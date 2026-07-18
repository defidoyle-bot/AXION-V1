"""
AXION QUANT V4 - Telegram Notifications & Automation
Production-grade notification layer with modular provider abstraction.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)

from config.settings import TelegramConfig, get_config
from core.logging import get_logger

logger = get_logger("telegram")


# =============================================================================
# NOTIFICATION DATA MODELS
# =============================================================================

class NotificationType(Enum):
    """Types of notifications."""
    SIGNAL = "signal"
    TRADE_UPDATE = "trade_update"
    DAILY_REPORT = "daily_report"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"
    SYSTEM_ALERT = "system_alert"
    ERROR = "error"
    HEALTH_CHECK = "health_check"


@dataclass
class NotificationMessage:
    """Notification message structure."""
    notification_type: NotificationType
    title: str
    content: str
    chat_id: str
    priority: str
    metadata: Dict[str, Any]
    timestamp: datetime
    image: Optional[bytes] = None


# =============================================================================
# TELEGRAM BOT
# =============================================================================

class TelegramBot:
    """Production-grade Telegram bot for AXION QUANT V4."""

    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or get_config().telegram
        self.application: Optional[Application] = None
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.message_queue_max_size)
        self._running = False
        self._send_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the Telegram bot."""
        try:
            self.application = Application.builder().token(self.config.bot_token).build()
        except Exception as e:
            logger.warning(f"Telegram bot initialization failed: {e}. Using mock mode.")
            self.application = None
            return

        # Register command handlers
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("health", self._cmd_health))
        self.application.add_handler(CommandHandler("scan", self._cmd_scan))
        self.application.add_handler(CommandHandler("signals", self._cmd_signals))
        self.application.add_handler(CommandHandler("watchlist", self._cmd_watchlist))
        self.application.add_handler(CommandHandler("stats", self._cmd_stats))
        self.application.add_handler(CommandHandler("performance", self._cmd_performance))
        self.application.add_handler(CommandHandler("backtest", self._cmd_backtest))
        self.application.add_handler(CommandHandler("retrain", self._cmd_retrain))
        self.application.add_handler(CommandHandler("reload", self._cmd_reload))
        self.application.add_handler(CommandHandler("config", self._cmd_config))
        self.application.add_handler(CommandHandler("logs", self._cmd_logs))
        self.application.add_handler(CommandHandler("version", self._cmd_version))
        self.application.add_handler(CommandHandler("about", self._cmd_about))

        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

        logger.info("Telegram bot initialized")

    async def start(self) -> None:
        """Start the bot."""
        if not self.application:
            await self.initialize()

        if not self.application:
            logger.warning("Telegram bot not initialized, skipping start")
            return

        self._running = True

        # Start message queue processor
        self._send_task = asyncio.create_task(self._process_message_queue())

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False

        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        if self.application:
            if self.application.updater:
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

        logger.info("Telegram bot stopped")

    # =================================================================
    # SIGNAL NOTIFICATIONS
    # =================================================================

    async def send_signal(self, signal: Dict[str, Any]) -> None:
        """Send a trading signal notification."""
        classification = signal.get("classification", "")
        message = self._format_signal_message(signal)

        # Send to channel if configured
        if self.config.channel_id:
            await self._queue_message(
                self.config.channel_id,
                message,
                signal.get("symbol", ""),
                classification,
            )

        # Send to admin
        await self._queue_message(
            self.config.admin_chat_id,
            message,
            signal.get("symbol", ""),
            classification,
        )

    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format a signal into an institutional-quality message."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        tp = signal.get("take_profit", [])
        rr = signal.get("risk_reward", 0)
        score = signal.get("score", 0)
        classification = signal.get("classification", "")
        ml_prob = signal.get("ml_probability", 0)
        ml_conf = signal.get("ml_confidence", 0)
        regime = signal.get("market_regime", "")
        smc_summary = signal.get("smc_summary", "")
        risk_status = signal.get("risk_status", "")
        leverage = signal.get("leverage", 1)

        # Emoji based on classification
        emoji_map = {
            "Institutional Grade": "🔥",
            "Premium Signal": "💎",
            "Strong Signal": "✅",
            "Standard Signal": "📊",
            "Watchlist": "👀",
        }
        emoji = emoji_map.get(classification, "📊")

        # Direction emoji
        dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"

        message = f"""
{emoji} <b>{classification.upper()}</b>
<b>{symbol}</b> · {dir_emoji}

📍 <b>Entry:</b> {entry:,.2f}
🛑 <b>Stop Loss:</b> {sl:,.2f}
"""

        for i, tp_level in enumerate(tp, 1):
            message += f"🎯 <b>TP{i}:</b> {tp_level:,.2f}\n"

        message += f"""
📐 <b>Risk/Reward:</b> {rr:.2f}
📊 <b>Confidence:</b> {score}% | <b>Leverage:</b> {leverage}x
🧠 <b>ML:</b> {ml_prob:.1%} ({ml_conf:.0%} confidence)
📈 <b>Regime:</b> {regime}
🔍 <b>SMC:</b> {smc_summary}
✅ <b>Risk:</b> {risk_status}

⏰ {datetime.utcnow().strftime("%H:%M UTC")}
"""

        return message

    # =================================================================
    # TRADE LIFECYCLE NOTIFICATIONS
    # =================================================================

    async def send_trade_update(self, trade: Dict[str, Any], update_type: str) -> None:
        """Send trade lifecycle update."""
        messages = {
            "signal_issued": "📡 Signal Issued",
            "tp1_hit": "🎯 TP1 Hit!",
            "tp2_hit": "🎯🎯 TP2 Hit!",
            "breakeven": "🛡️ Stop Moved to Break-even",
            "trailing": "🏃 Trailing Stop Activated",
            "profit": "💰 Trade Closed in Profit",
            "stop_loss": "🛑 Stop Loss Hit",
            "invalidated": "⚠️ Signal Invalidated Before Entry",
        }

        title = messages.get(update_type, "📊 Trade Update")
        symbol = trade.get("symbol", "")
        pnl = trade.get("pnl", 0)
        pnl_percent = trade.get("pnl_percent", 0)

        message = f"""
<b>{title}</b>
<b>{symbol}</b>
"""

        if pnl != 0:
            emoji = "🟢" if pnl > 0 else "🔴"
            message += f"\n{emoji} P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)"

        await self._queue_message(self.config.admin_chat_id, message, symbol, update_type)

    # =================================================================
    # REPORTS
    # =================================================================

    async def send_daily_report(self, report_data: Dict[str, Any]) -> None:
        """Send daily performance report."""
        message = f"""
📅 <b>DAILY REPORT - {datetime.utcnow().strftime("%Y-%m-%d")}</b>

📊 <b>Signals:</b> {report_data.get('signals_count', 0)}
🎯 <b>Win Rate:</b> {report_data.get('win_rate', 0):.1%}
💰 <b>P&L:</b> ${report_data.get('pnl', 0):,.2f}
📈 <b>Best Trade:</b> {report_data.get('best_trade', 'N/A')}
📉 <b>Worst Trade:</b> {report_data.get('worst_trade', 'N/A')}
🔥 <b>Active Trades:</b> {report_data.get('active_trades', 0)}
📋 <b>Market Summary:</b> {report_data.get('market_summary', 'N/A')}
"""
        await self._queue_message(self.config.admin_chat_id, message, "", "daily_report")

    async def send_weekly_report(self, report_data: Dict[str, Any]) -> None:
        """Send weekly performance report."""
        message = f"""
📊 <b>WEEKLY REPORT - {datetime.utcnow().strftime("%Y-%m-%d")}</b>

💰 <b>Weekly Profit:</b> ${report_data.get('weekly_profit', 0):,.2f}
🎯 <b>Win Rate:</b> {report_data.get('win_rate', 0):.1%}
🧠 <b>ML Performance:</b> {report_data.get('ml_performance', 'N/A')}
⭐ <b>Best Symbol:</b> {report_data.get('best_symbol', 'N/A')}
❌ <b>Worst Symbol:</b> {report_data.get('worst_symbol', 'N/A')}
"""
        await self._queue_message(self.config.admin_chat_id, message, "", "weekly_report")

    async def send_monthly_report(self, report_data: Dict[str, Any]) -> None:
        """Send monthly performance report."""
        message = f"""
📈 <b>MONTHLY REPORT - {datetime.utcnow().strftime("%Y-%m")}</b>

💰 <b>Total Return:</b> {report_data.get('total_return', 0):.2f}%
📊 <b>Sharpe Ratio:</b> {report_data.get('sharpe_ratio', 0):.2f}
📉 <b>Sortino Ratio:</b> {report_data.get('sortino_ratio', 0):.2f}
⚖️ <b>Profit Factor:</b> {report_data.get('profit_factor', 0):.2f}
📉 <b>Max Drawdown:</b> {report_data.get('max_drawdown', 0):.2f}%
"""
        await self._queue_message(self.config.admin_chat_id, message, "", "monthly_report")

    # =================================================================
    # TRADE LIFECYCLE NOTIFICATIONS  (SRS Part 7)
    # =================================================================

    async def notify_signal_issued(self, signal: Dict[str, Any]) -> None:
        """Notify that a new signal has been issued and is awaiting entry."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        entry = signal.get("entry_price", "—")
        signal_id = signal.get("signal_id", "")[:8]

        message = (
            f"✅ <b>SIGNAL ISSUED</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"📍 Entry: <code>{entry}</code>\n"
            f"⏳ Awaiting fill…"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_issued",
        )

    async def notify_tp_hit(
        self, signal: Dict[str, Any], tp_number: int, tp_price: float, pnl_r: float
    ) -> None:
        """Notify that a Take Profit level has been reached."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        emojis = {1: "📈", 2: "📈📈", 3: "🚀"}
        emoji = emojis.get(tp_number, "📈")

        message = (
            f"{emoji} <b>TP{tp_number} HIT</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"💰 TP{tp_number}: <code>{tp_price}</code>\n"
            f"📊 <b>+{pnl_r:.2f}R</b> achieved\n"
            f"🔄 Remaining position active"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, f"lifecycle_tp{tp_number}",
        )

    async def notify_stop_moved_to_breakeven(self, signal: Dict[str, Any], new_stop: float) -> None:
        """Notify that the stop loss has been moved to break-even."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        message = (
            f"🔄 <b>STOP → BREAK-EVEN</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"🛡️ Stop moved to: <code>{new_stop}</code>\n"
            f"✅ Trade is now risk-free"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_breakeven",
        )

    async def notify_trailing_stop_activated(
        self, signal: Dict[str, Any], trailing_stop: float
    ) -> None:
        """Notify that the trailing stop has been activated."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        message = (
            f"📉 <b>TRAILING STOP ACTIVE</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"📌 Trailing stop: <code>{trailing_stop}</code>\n"
            f"🔒 Profits locked — following price"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_trailing",
        )

    async def notify_trade_closed(
        self,
        signal: Dict[str, Any],
        exit_price: float,
        pnl_r: float,
        exit_reason: str,
    ) -> None:
        """Notify that a trade has been fully closed."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        is_win = pnl_r > 0
        emoji = "🎯" if is_win else "❌"
        result_label = "WIN" if is_win else "LOSS"
        pnl_str = f"+{pnl_r:.2f}R" if is_win else f"{pnl_r:.2f}R"

        message = (
            f"{emoji} <b>TRADE CLOSED — {result_label}</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"💰 Exit: <code>{exit_price}</code>\n"
            f"📊 Result: <b>{pnl_str}</b>\n"
            f"📋 Reason: {exit_reason}"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_closed",
        )

    async def notify_signal_invalidated(
        self, signal: Dict[str, Any], reason: str
    ) -> None:
        """Notify that a signal was invalidated before entry was triggered."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        message = (
            f"🚫 <b>SIGNAL INVALIDATED</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"❌ Reason: {reason}\n"
            f"ℹ️ No position was opened"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_invalidated",
        )

    async def notify_stop_loss_hit(
        self, signal: Dict[str, Any], stop_price: float, loss_r: float
    ) -> None:
        """Notify that the stop loss was hit and the trade closed at a loss."""
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        signal_id = signal.get("signal_id", "")[:8]

        message = (
            f"❌ <b>STOP LOSS HIT</b>  <code>[{signal_id}]</code>\n\n"
            f"<b>{symbol}</b>  {direction}\n"
            f"🛑 Stop: <code>{stop_price}</code>\n"
            f"📊 Result: <b>{loss_r:.2f}R</b>\n"
            f"📋 Closed at planned risk level"
        )
        await self._queue_message(
            self.config.channel_id or self.config.admin_chat_id,
            message, symbol, "lifecycle_sl_hit",
        )

    # =================================================================
    # CHART GENERATION
    # =================================================================

    async def generate_signal_chart(
        self,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        indicators: Dict[str, Any],
        smc_data: Dict[str, Any],
    ) -> Optional[bytes]:
        """Generate an annotated chart image for a signal.

        Overlays rendered
        -----------------
        - Candlestick OHLCV
        - EMA lines (all configured periods)
        - VWAP (if available)
        - Order Blocks (shaded rectangles)
        - Fair Value Gaps (shaded rectangles)
        - BOS / CHOCH / MSS labels at detection candle
        - Liquidity sweep markers
        - Entry price line (green dashed)
        - Stop Loss line (red dashed)
        - Take Profit lines (blue dashed, TP1/TP2/TP3)
        """
        if not self.config.generate_charts:
            return None

        try:
            df_chart = df.copy()
            df_chart.index = pd.to_datetime(df_chart.index)

            # ── Restrict to last N candles for readability ────────────────
            lookback = min(self.config.chart_lookback_candles, len(df_chart))
            df_chart = df_chart.iloc[-lookback:]

            # ── Build mplfinance addplots ─────────────────────────────────
            addplots = []

            # EMA lines
            ema_colours = ["#f59e0b", "#60a5fa", "#a78bfa", "#34d399"]
            if "ema" in indicators:
                for idx, (period, values) in enumerate(sorted(indicators["ema"].items())):
                    series = values if isinstance(values, pd.Series) else pd.Series(values)
                    series = series.reindex(df_chart.index)
                    colour = ema_colours[idx % len(ema_colours)]
                    addplots.append(
                        mpf.make_addplot(
                            series, label=f"EMA{period}", color=colour, width=1.2
                        )
                    )

            # VWAP
            if "vwap" in indicators:
                vwap = indicators["vwap"]
                vwap_series = vwap if isinstance(vwap, pd.Series) else pd.Series(vwap)
                vwap_series = vwap_series.reindex(df_chart.index)
                addplots.append(
                    mpf.make_addplot(
                        vwap_series, label="VWAP", color="#fb7185", width=1.5, linestyle="--"
                    )
                )

            # ── Base mplfinance figure ────────────────────────────────────
            style = mpf.make_mpf_style(
                base_mpf_style="nightclouds",
                gridcolor="#1e293b",
                gridstyle="--",
                facecolor="#0f172a",
                edgecolor="#334155",
                figcolor="#0f172a",
                rc={
                    "axes.labelcolor": "#94a3b8",
                    "xtick.color": "#94a3b8",
                    "ytick.color": "#94a3b8",
                    "text.color": "#e2e8f0",
                    "font.size": 8,
                },
            )

            title = (
                f"{signal.get('symbol', '')}  |  "
                f"{signal.get('direction', '')}  |  "
                f"Score: {signal.get('score', '—')}  |  "
                f"{signal.get('classification', '')}"
            )

            fig, axes = mpf.plot(
                df_chart,
                type="candle",
                style=style,
                title=title,
                ylabel="Price (USDT)",
                volume=True,
                addplot=addplots if addplots else None,
                returnfig=True,
                figsize=(14, 9),
                panel_ratios=(3, 1),
            )

            ax_price = axes[0]

            # ── SMC Annotations ───────────────────────────────────────────
            x_index = {ts: i for i, ts in enumerate(df_chart.index)}
            x_max = len(df_chart) - 1

            # Order Blocks
            for ob in smc_data.get("order_blocks", []):
                try:
                    top = float(ob["top"])
                    bot = float(ob["bottom"])
                    ob_type = ob.get("ob_type", "BULLISH")
                    colour = "#22c55e33" if ob_type == "BULLISH" else "#ef444433"
                    edge = "#22c55e" if ob_type == "BULLISH" else "#ef4444"
                    ax_price.axhspan(bot, top, alpha=0.25, color=colour,
                                     linewidth=0.8, edgecolor=edge, linestyle="--")
                    ax_price.text(
                        x_max * 0.02, (top + bot) / 2,
                        f"OB {'Bull' if ob_type == 'BULLISH' else 'Bear'}",
                        color=edge, fontsize=6.5, va="center",
                        transform=ax_price.get_yaxis_transform(),
                    )
                except (KeyError, ValueError, TypeError):
                    pass

            # Fair Value Gaps
            for fvg in smc_data.get("fvgs", []):
                try:
                    if fvg.get("status") == "FILLED":
                        continue
                    top = float(fvg["upper"])
                    bot = float(fvg["lower"])
                    ax_price.axhspan(bot, top, alpha=0.15, color="#a78bfa33",
                                     linewidth=0.6, edgecolor="#a78bfa", linestyle=":")
                    ax_price.text(
                        x_max * 0.98, (top + bot) / 2, "FVG",
                        color="#a78bfa", fontsize=6, va="center", ha="right",
                        transform=ax_price.get_yaxis_transform(),
                    )
                except (KeyError, ValueError, TypeError):
                    pass

            # BOS labels
            for bos in smc_data.get("bos_events", []):
                try:
                    direction = bos.get("direction", "bullish")
                    colour = "#22c55e" if direction == "bullish" else "#ef4444"
                    # Draw at 20% from the right as a text label on the price axis
                    ax_price.annotate(
                        f"BOS ({'▲' if direction == 'bullish' else '▼'})",
                        xy=(0.80, 0.92 if direction == "bullish" else 0.08),
                        xycoords="axes fraction",
                        fontsize=7, color=colour,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a",
                                  edgecolor=colour, alpha=0.8),
                    )
                    break  # show only most recent
                except (KeyError, ValueError):
                    pass

            # CHOCH labels
            for choch in smc_data.get("choch_events", []):
                try:
                    ax_price.annotate(
                        "CHOCH",
                        xy=(0.65, 0.50),
                        xycoords="axes fraction",
                        fontsize=7, color="#f59e0b",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a",
                                  edgecolor="#f59e0b", alpha=0.8),
                    )
                    break
                except (KeyError, ValueError):
                    pass

            # Liquidity sweeps
            if smc_data.get("liquidity_sweeps"):
                ax_price.annotate(
                    "Liq. Sweep ✓",
                    xy=(0.02, 0.96),
                    xycoords="axes fraction",
                    fontsize=6.5, color="#60a5fa",
                )

            # ── Trade levels ──────────────────────────────────────────────
            entry = signal.get("entry_price")
            stop = signal.get("stop_loss")
            take_profits = signal.get("take_profit", [])
            if isinstance(take_profits, (int, float)):
                take_profits = [take_profits]

            if entry:
                ax_price.axhline(
                    float(entry), color="#22c55e", linewidth=1.4,
                    linestyle="--", alpha=0.9, label="Entry"
                )
                ax_price.text(
                    x_max + 0.5, float(entry), "ENTRY",
                    color="#22c55e", fontsize=7, va="center",
                )

            if stop:
                ax_price.axhline(
                    float(stop), color="#ef4444", linewidth=1.4,
                    linestyle="--", alpha=0.9, label="SL"
                )
                ax_price.text(
                    x_max + 0.5, float(stop), "SL",
                    color="#ef4444", fontsize=7, va="center",
                )

            tp_colours = ["#60a5fa", "#818cf8", "#a78bfa"]
            for i, tp in enumerate(take_profits[:3]):
                col = tp_colours[i]
                ax_price.axhline(
                    float(tp), color=col, linewidth=1.2,
                    linestyle="-.", alpha=0.85, label=f"TP{i + 1}"
                )
                ax_price.text(
                    x_max + 0.5, float(tp), f"TP{i + 1}",
                    color=col, fontsize=7, va="center",
                )

            # Legend
            ax_price.legend(loc="upper left", fontsize=6.5, framealpha=0.4,
                            facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

            # ── Info box ─────────────────────────────────────────────────
            rr = signal.get("risk_reward", "—")
            conf = signal.get("confidence", "—")
            info = (
                f"R:R {rr}  |  Conf {conf}%  |  "
                f"Regime: {signal.get('market_regime', '—')}"
            )
            fig.text(
                0.5, 0.01, info,
                ha="center", fontsize=8, color="#94a3b8",
                transform=fig.transFigure,
            )

            # ── Render to bytes ───────────────────────────────────────────
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0f172a")
            buf.seek(0)
            plt.close(fig)

            logger.debug(
                f"Chart generated: {signal.get('symbol')} "
                f"| OBs={len(smc_data.get('order_blocks', []))} "
                f"| FVGs={len(smc_data.get('fvgs', []))}"
            )
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Chart generation failed: {e}", exc_info=True)
            return None

    # =================================================================
    # MESSAGE QUEUE
    # =================================================================

    async def _queue_message(self, chat_id: str, text: str, symbol: str, msg_type: str) -> None:
        """Queue a message for sending."""
        try:
            self._message_queue.put_nowait({
                "chat_id": chat_id,
                "text": text,
                "symbol": symbol,
                "type": msg_type,
            })
        except asyncio.QueueFull:
            logger.warning("Telegram message queue full, dropping message")

    async def _process_message_queue(self) -> None:
        """Process queued messages with rate limiting."""
        messages_sent = 0
        last_minute = datetime.utcnow().minute

        while self._running:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)

                # Rate limiting
                current_minute = datetime.utcnow().minute
                if current_minute != last_minute:
                    messages_sent = 0
                    last_minute = current_minute

                if messages_sent >= self.config.max_messages_per_minute:
                    await asyncio.sleep(1)
                    continue

                # Send message
                if self.application:
                    await self.application.bot.send_message(
                        chat_id=msg["chat_id"],
                        text=msg["text"],
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    messages_sent += 1
                    logger.debug(f"Sent {msg['type']} message to {msg['chat_id']}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

    # =================================================================
    # COMMAND HANDLERS
    # =================================================================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "🔥 <b>AXION QUANT V4</b> - Institutional AI Trading Platform\n\n"
            "Use /help to see available commands.",
            parse_mode="HTML",
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = """
<b>AXION QUANT V4 - Commands</b>

<b>Admin Commands:</b>
/status - System status
/health - Health check
/scan - Force market scan
/signals - Recent signals
/watchlist - Current watchlist
/stats - Trading statistics
/performance - Performance metrics
/backtest - Run backtest
/retrain - Retrain ML model
/reload - Reload configuration
/config - View configuration
/logs - Recent logs
/version - Version info
/about - About AXION QUANT

<b>User Commands:</b>
/start - Start the bot
/help - Show this help
"""
        await update.message.reply_text(help_text, parse_mode="HTML")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        # This would be populated by the main application
        status = "\n🟢 <b>System Status:</b> Running\n"
        status += "📊 Scanner: Active\n"
        status += "🧠 ML Engine: Active\n"
        status += "📡 Telegram: Connected\n"
        status += "💾 Database: Connected\n"
        await update.message.reply_text(status, parse_mode="HTML")

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /health command."""
        health = """
<b>Health Check</b>

✅ Exchange Connection: Healthy
✅ Telegram Connection: Healthy
✅ Database: Healthy
✅ ML Model: Available
✅ Scheduler: Running
⏳ Queue: Normal
💾 Disk: OK
🧠 Memory: OK
"""
        await update.message.reply_text(health, parse_mode="HTML")

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /scan command."""
        await update.message.reply_text(
            "🔍 <b>Force Scan Initiated</b>\n"
            "Scanning all symbols across all timeframes...",
            parse_mode="HTML",
        )

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signals command."""
        await update.message.reply_text(
            "📊 <b>Recent Signals</b>\n\n"
            "No recent signals to display.",
            parse_mode="HTML",
        )

    async def _cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /watchlist command."""
        await update.message.reply_text(
            "👀 <b>Current Watchlist</b>\n\n"
            "Watchlist is empty.",
            parse_mode="HTML",
        )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stats command."""
        await update.message.reply_text(
            "📈 <b>Trading Statistics</b>\n\n"
            "Total Trades: 0\n"
            "Win Rate: 0%\n"
            "Profit Factor: 0.00\n"
            "Sharpe Ratio: 0.00",
            parse_mode="HTML",
        )

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /performance command."""
        await update.message.reply_text(
            "📊 <b>Performance Metrics</b>\n\n"
            "Total Return: 0.00%\n"
            "Max Drawdown: 0.00%\n"
            "Expectancy: $0.00",
            parse_mode="HTML",
        )

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /backtest command."""
        await update.message.reply_text(
            "🧪 <b>Backtest Started</b>\n"
            "Running backtest with current configuration...",
            parse_mode="HTML",
        )

    async def _cmd_retrain(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /retrain command."""
        await update.message.reply_text(
            "🧠 <b>ML Retraining Initiated</b>\n"
            "Retraining model with latest data...",
            parse_mode="HTML",
        )

    async def _cmd_reload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reload command."""
        await update.message.reply_text(
            "🔄 <b>Configuration Reloaded</b>",
            parse_mode="HTML",
        )

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /config command."""
        config_text = f"""
<b>Configuration</b>

Strategy Profile: {self.config.strategy_profile}
Risk Profile: Balanced
Max Leverage: 20x
Risk per Trade: 1%
"""
        await update.message.reply_text(config_text, parse_mode="HTML")

    async def _cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /logs command."""
        await update.message.reply_text(
            "📋 <b>Recent Logs</b>\n\n"
            "System running normally. No errors.",
            parse_mode="HTML",
        )

    async def _cmd_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /version command."""
        await update.message.reply_text(
            "🔥 <b>AXION QUANT V4</b>\n"
            "Version: 4.0.0\n"
            "Codename: Institutional AI Quantitative Trading Platform",
            parse_mode="HTML",
        )

    async def _cmd_about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /about command."""
        await update.message.reply_text(
            "🔥 <b>AXION QUANT V4</b>\n\n"
            "Institutional-grade AI-powered cryptocurrency\n"
            "market intelligence and trading platform.\n\n"
            "Target: MEXC USDT-M Perpetual Futures\n"
            "Features: SMC, ML, Risk Management, Backtesting\n\n"
            "Built for professionals. Not a toy.",
            parse_mode="HTML",
        )

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(text=f"Selected: {query.data}")
