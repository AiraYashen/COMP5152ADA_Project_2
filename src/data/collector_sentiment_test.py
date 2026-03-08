import yfinance as yf

# 1. 测试指数 ^NDX（纳斯达克100）
ndx = yf.Ticker("^NDX")
ndx_news = ndx.news
print("=== ^NDX 新闻 ===")
print(f"类型: {type(ndx_news)}")
print(f"内容: {ndx_news}")
print(f"是否为空: {ndx_news == [] or ndx_news is None}")
print("-" * 50)

# 2. 测试知名个股（苹果，应该有新闻）
aapl = yf.Ticker("AAPL")
aapl_news = aapl.news
print("=== AAPL 新闻 ===")
print(f"类型: {type(aapl_news)}")
print(f"内容: {aapl_news}")
print(f"是否为空: {aapl_news == [] or aapl_news is None}")
print("-" * 50)

# 3. 测试另一个指数（标普500，对比用）
spx = yf.Ticker("^GSPC")
spx_news = spx.news
print("=== ^GSPC 新闻 ===")
print(f"类型: {type(spx_news)}")
print(f"内容: {spx_news}")
print(f"是否为空: {spx_news == [] or spx_news is None}")