"""
Main execution script for Cryptocurrency Analysis Tool
"""

from src.crypto_analyzer import CryptoAnalyzer
from src.history_manager import HistoryManager
from src.plotter import CryptoPlotter
import sys


def display_menu():
    """Display main menu options"""
    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY ANALYSIS TOOL")
    print("=" * 60)
    print("1. Analyze top cryptocurrencies")
    print("2. View analysis history")
    print("3. Compare multiple coins")
    print("4. Generate advanced charts")
    print("5. Exit")
    print("=" * 60)


def analyze_top_coins():
    """Analyze and display top cryptocurrencies"""
    analyzer = CryptoAnalyzer()
    plotter = CryptoPlotter()
    history = HistoryManager()
    
    try:
        # Get top coins
        print("\n🔄 Fetching top cryptocurrencies...")
        top_coins = analyzer.get_top_coins(20)
        
        if not top_coins:
            print("❌ Failed to fetch top coins")
            return
        
        # Display top coins
        print("\n📊 TOP 20 CRYPTOCURRENCIES BY MARKET CAP")
        print("-" * 70)
        for i, coin in enumerate(top_coins, 1):
            print(f"{i:2}. {coin['name']:20} ({coin['symbol'].upper():5}) - "
                  f"${coin['current_price']:12,.2f} | "
                  f"24h: {coin['price_change_percentage_24h']:6.2f}%")
        print("-" * 70)
        
        # User selection
        while True:
            try:
                choice = int(input("\n🔍 Enter coin number to analyze (1-20): "))
                if 1 <= choice <= 20:
                    selected_coin = top_coins[choice - 1]
                    break
                print("❌ Please enter a number between 1 and 20")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get days for analysis
        days = int(input("📅 Days of historical data (default 30): ") or 30)
        days = max(1, min(365, days))  # Clamp between 1 and 365
        
        print(f"\n📈 Analyzing {selected_coin['name']} ({selected_coin['symbol'].upper()})...")
        
        # Fetch historical data
        df = analyzer.get_coin_data(selected_coin['id'], days)
        if df is None:
            print("❌ Failed to fetch historical data")
            return
        
        # Calculate technical indicators
        df = analyzer.calculate_technical_indicators(df)
        if df is None:
            print("❌ Failed to calculate indicators")
            return
        
        # Generate predictions and signals
        predictions = analyzer.generate_predictions(df)
        signal = analyzer.get_trading_signal(df)
        current_price = analyzer.get_current_price(selected_coin['id'])
        
        # Display results
        print("\n" + "=" * 60)
        print(f"📊 ANALYSIS RESULTS - {selected_coin['name'].upper()}")
        print("=" * 60)
        
        print(f"\n💰 CURRENT METRICS:")
        print(f"   • Price: ${current_price:,.2f}")
        print(f"   • Market Cap: ${selected_coin['market_cap']:,.0f}")
        print(f"   • 24h Volume: ${selected_coin['total_volume']:,.0f}")
        print(f"   • 24h Change: {selected_coin['price_change_percentage_24h']:.2f}%")
        
        print(f"\n📊 TECHNICAL INDICATORS:")
        if 'rsi' in df and not df['rsi'].isna().all():
            rsi = df['rsi'].iloc[-1]
            rsi_status = "🟢 Oversold (Bullish)" if rsi < 30 else "🔴 Overbought (Bearish)" if rsi > 70 else "⚪ Neutral"
            print(f"   • RSI (14): {rsi:.1f} - {rsi_status}")
        
        if 'macd_diff' in df and not df['macd_diff'].isna().all():
            macd = df['macd_diff'].iloc[-1]
            macd_status = "🟢 Bullish" if macd > 0 else "🔴 Bearish"
            print(f"   • MACD: {macd:.2f} - {macd_status}")
        
        if 'ema_20' in df and not df['ema_20'].isna().all():
            ema = df['ema_20'].iloc[-1]
            price_vs_ema = (current_price / ema - 1) * 100
            print(f"   • EMA 20: ${ema:,.2f} (Price {price_vs_ema:+.1f}% from EMA)")
        
        print(f"\n🎯 TRADING SIGNAL: ", end="")
        if signal == 'BUY':
            print("🟢 BUY - Bullish indicators suggest upward movement")
        elif signal == 'SELL':
            print("🔴 SELL - Bearish indicators suggest downward movement")
        else:
            print("⚪ HOLD - Mixed signals, wait for clearer direction")
        
        print(f"\n📈 PRICE PREDICTIONS:")
        print(f"   • 1-Day:  BUY {predictions['1_day']['buy']}% | "
              f"SELL {predictions['1_day']['sell']}% | "
              f"HOLD {predictions['1_day']['hold']}%")
        print(f"   • 5-Day:  BUY {predictions['5_day']['buy']}% | "
              f"SELL {predictions['5_day']['sell']}% | "
              f"HOLD {predictions['5_day']['hold']}%")
        
        # Save to history
        history_entry = {
            'coin_id': selected_coin['id'],
            'coin_name': selected_coin['name'],
            'symbol': selected_coin['symbol'],
            'price': current_price,
            'signal': signal,
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df else None,
            'predictions': predictions,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        history.save_analysis(history_entry)
        
        # Ask for plotting
        plot_choice = input("\n📊 Generate technical charts? (y/n): ").strip().lower()
        if plot_choice == 'y':
            print("🔄 Generating charts...")
            plotter.plot_full_analysis(df, selected_coin['name'], predictions, 
                                      f"{selected_coin['id']}_analysis.png")
            plotter.show()
            
        # Ask for candlestick chart
        candle_choice = input("\n🕯️ Generate advanced candlestick chart? (y/n): ").strip().lower()
        if candle_choice == 'y':
            print("🔄 Generating candlestick chart...")
            plotter.plot_advanced_candlestick(df, selected_coin['name'])
            plotter.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def view_history():
    """View analysis history"""
    history = HistoryManager()
    
    stats = history.get_statistics()
    print("\n" + "=" * 60)
    print("📜 ANALYSIS HISTORY")
    print("=" * 60)
    print(f"📊 Total analyses: {stats['total_analyses']}")
    print(f"🪙 Unique coins: {stats['unique_coins']}")
    print(f"🟢 Buy signals: {stats['buy_signals']}")
    print(f"🔴 Sell signals: {stats['sell_signals']}")
    print(f"⚪ Hold signals: {stats['hold_signals']}")
    
    view_all = input("\n📋 View recent analyses? (y/n): ").strip().lower()
    if view_all == 'y':
        history_data = history.get_history(limit=10)
        if not history_data:
            print("No analysis history found")
        else:
            print("\nRecent Analyses:")
            print("-" * 60)
            for entry in history_data:
                print(f"• {entry['coin_name']} ({entry['symbol'].upper()}): "
                      f"{entry['signal']} @ ${entry['price']:,.2f}")
                print(f"  {entry['timestamp'][:19]}")
                print()


def compare_coins():
    """Compare multiple cryptocurrencies"""
    analyzer = CryptoAnalyzer()
    plotter = CryptoPlotter()
    
    print("\n" + "=" * 60)
    print("🔄 COIN COMPARISON")
    print("=" * 60)
    
    coins_input = input("Enter coin IDs separated by commas (e.g., bitcoin,ethereum,solana): ")
    coins = [c.strip().lower() for c in coins_input.split(',')]
    days = int(input("Days of data for comparison (default 30): ") or 30)
    
    comparison_data = {}
    
    for coin in coins:
        try:
            print(f"🔄 Fetching data for {coin}...")
            df = analyzer.get_coin_data(coin, days)
            if df is not None:
                comparison_data[coin] = df['close']
        except Exception as e:
            print(f"❌ Failed to fetch {coin}: {e}")
    
    if comparison_data:
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize to percentage change
        for coin, prices in comparison_data.items():
            normalized = (prices / prices.iloc[0] - 1) * 100
            ax.plot(normalized.index, normalized, label=coin.capitalize(), linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cryptocurrency Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("coin_comparison.png", dpi=150)
        plt.show()
        print("✓ Comparison chart saved to coin_comparison.png")


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("🚀 CRYPTOCURRENCY TECHNICAL ANALYSIS TOOL")
    print("   Powered by CoinGecko API & TA-Lib")
    print("=" * 60)
    
    while True:
        display_menu()
        choice = input("\n👉 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            analyze_top_coins()
        elif choice == '2':
            view_history()
        elif choice == '3':
            compare_coins()
        elif choice == '4':
            analyze_top_coins()  # This already includes chart generation
        elif choice == '5':
            print("\n👋 Thank you for using Crypto Analysis Tool!")
            print("📊 Happy Trading!\n")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please enter 1-5")


if __name__ == "__main__":
    main()