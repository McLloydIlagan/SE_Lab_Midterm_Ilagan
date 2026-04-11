import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher
from src.indicator_calculator import IndicatorCalculator
from src.price_predictor import PricePredictor
from src.backtest_engine import BacktestEngine
from src.visualizer import Visualizer


def main():
    print("=" * 60)
    print("🚀 CRYPTOCURRENCY TECHNICAL ANALYSIS TOOL")
    print("   Powered by CoinGecko API & Machine Learning")
    print("=" * 60)
    
    # Initialize modules
    fetcher = DataFetcher()
    calculator = IndicatorCalculator()
    predictor = PricePredictor()
    
    try:
        # Get top coins
        print("\n📊 Fetching top cryptocurrencies...")
        top_coins = fetcher.get_top_coins(10)
        
        print("\n🔝 TOP 10 CRYPTOCURRENCIES:")
        print("-" * 50)
        for i, coin in enumerate(top_coins, 1):
            print(f"{i:2}. {coin['name']:20} ${coin['current_price']:12,.2f}")
        print("-" * 50)
        
        # User selection
        choice = int(input("\n👉 Select coin (1-10): ")) - 1
        selected_coin = top_coins[choice]
        
        print(f"\n📈 Analyzing {selected_coin['name']}...")
        
        # Fetch historical data
        df = fetcher.fetch_ohlcv(selected_coin['id'], days=100)
        
        # Calculate indicators
        df = calculator.compute_indicators(df)
        
        # Display results
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        
        print("\n" + "=" * 50)
        print("📊 TECHNICAL ANALYSIS RESULTS")
        print("=" * 50)
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📊 RSI (14): {current_rsi:.1f}")
        print(f"   Signal: {calculator.get_rsi_signal(current_rsi)}")
        print(f"📈 MACD: {current_macd:.3f}")
        print(f"   Signal: {calculator.get_macd_signal(current_macd)}")
        
        # Make predictions
        pred_1d = predictor.predict_price(df, days_ahead=1)
        pred_5d = predictor.predict_price(df, days_ahead=5)
        
        print(f"\n🔮 PRICE PREDICTIONS:")
        print(f"   1-Day: ${pred_1d:,.2f}")
        print(f"   5-Day: ${pred_5d:,.2f}")
        
        # Overall recommendation
        if pred_1d > current_price:
            print("\n✅ RECOMMENDATION: BUY - Price expected to increase")
        else:
            print("\n❌ RECOMMENDATION: SELL - Price expected to decrease")
        
        # Ask for charts
        show_charts = input("\n📊 Generate charts? (y/n): ").lower()
        if show_charts == 'y':
            visualizer = Visualizer()
            print("Generating charts...")
            visualizer.plot_technical_indicators(df, selected_coin['name'])
            visualizer.show()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tip: Make sure you have internet connection")
        print("   CoinGecko API might be rate limiting. Try again in a few seconds.")


if __name__ == "__main__":
    main()