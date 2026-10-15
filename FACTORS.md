# What could make a memecoin run: 300 candidate factors

A catalogue for the pump watcher (`watcher.py`, `enrich.py`, `watch_features.py`). Most of
these will turn out to be noise. The point of the week-long watch is to let held-out data
decide, not intuition. Status as of 2026-10-15 (after building everything buildable):

- ✅ built: recorded by the watcher and computed for the analysis
- 🔴 not buildable with the data available, with the reason

A factor only helps if it is **known before the move**. Anything measured after the
snapshot is a leak, however predictive it looks.

---

## A. Price path and momentum (1–25)

1. ✅ Price change in the first 5 minutes
2. ✅ Price change over the last hour
3. ✅ Price now vs first-seen price
4. ✅ Price now vs the previous snapshot (acceleration between snapshots)
5. ✅ Drawdown from the local high before the snapshot (dip-buy setups)
6. ✅ Number of higher lows in the first 15 minutes (staircase vs spike)
7. ✅ Volatility of 1-minute returns before the snapshot (needs candles fetched at snapshot time)
8. ✅ Longest run of green 1-minute candles
9. ✅ Size of the largest single-minute drop (one big seller?)
10. ✅ Time since the last new high
11. ✅ Whether price is above its own VWAP since launch
12. ✅ Market cap vs pump.fun all-time high at the snapshot
13. ✅ Round-number market caps crossed ($10k, $30k, $69k, $100k): psychological levels
14. ✅ Distance to pump.fun's graduation market cap (tokens near graduation get bid up)
15. ✅ Speed of the first 2x from launch (minutes)
16. ✅ How many times it already doubled and halved (chop vs trend)
17. ✅ Price recovery after the first dump (resilience)
18. ✅ Candle wick ratio (long upper wicks = selling into strength)
19. ✅ Gap between DexScreener and Jupiter price (stale vs live demand)
20. ✅ Relative strength vs the median new token in the same hour
21. ✅ Relative strength vs SOL over the same window
22. ✅ Momentum decay: m5 change vs h1 change
23. ✅ Late-hour surge: last 5 minutes' share of the hour's move
24. ✅ Price stability after graduation (first 10 minutes on PumpSwap)
25. ✅ Whether it already had a failed pump (previous peak and dump) before the snapshot

## B. Volume and trade flow (26–45)

26. ✅ Buys and sells in the last 5 minutes
27. ✅ Buys and sells in the last hour
28. ✅ Unique buyers and sellers (5 minutes, 1 hour)
29. ✅ Buy ratio (buys / all trades)
30. ✅ Buyer ratio (unique buyers / all traders)
31. ✅ Trades per trader (high = bots churning)
32. ✅ Average trade size
33. ✅ Volume / liquidity (turnover)
34. ✅ Growth in unique buyers between snapshots
35. ✅ Median trade size (average hides whales vs retail)
36. ✅ Largest single buy in the window
37. ✅ Share of volume from the top 5 wallets
38. ✅ Net buy volume (buy $ − sell $), not just counts
39. ✅ Buy-size distribution shape (many small = retail; few big = insiders)
40. ✅ New buyers per minute, trend over the first 15 minutes
41. ✅ Repeat buyers (same wallet buying again: conviction or a bot ladder)
42. ✅ Sell pressure from early buyers (snipers taking profit)
43. ✅ Volume spike vs the token's own previous 5 minutes
44. ✅ Volume vs the median new token's volume at the same age
45. ✅ Time between trades (bursty bots vs steady humans)

## C. Liquidity and pool mechanics (46–60)

46. ✅ Liquidity in USD
47. ✅ Liquidity / FDV
48. ✅ Exchange: pump.fun curve, PumpSwap, Meteora, Raydium, other
49. ✅ Launchpad vs regular AMM
50. ✅ Number of pools for the token (more pools = more routing, more arbitrage)
51. ✅ Quote token: SOL vs USDC vs other
52. ✅ LP tokens burned vs locked vs held by a wallet (after graduation)
53. ✅ Liquidity added or removed since launch
54. ✅ Price impact of a $500 buy (Jupiter quote): how thin it is
55. ✅ Round-trip cost of $100 (Jupiter buy then sell quote)
56. ✅ Routable on Jupiter at all (aggregator flow)
57. ✅ On Jupiter's verified or strict token list
58. 🔴 Meteora dynamic fee level (fee spikes with volatility) — *needs decoding Meteora pool accounts per program; not built*
59. 🔴 Concentrated-liquidity range placement (Orca/Raydium CLMM pools) — *needs decoding concentrated-liquidity positions; not built*
60. ✅ Time from graduation to first $10k of AMM volume

## D. Bonding curve and launchpad (61–75)

61. ✅ Bonding-curve progress (graduation %)
62. ✅ Graduated or not
63. ✅ Time from launch to graduation
64. ✅ Curve progress per minute (velocity)
65. ✅ Launchpad: pump.fun, LetsBonk, Bags, Meteora DBC, Moonit, others
66. ✅ pump.fun livestream running
67. ✅ pump.fun boost mode
68. ✅ pump.fun security verdict
69. ✅ Creator cashback enabled
70. ✅ Tokenised AI agent flag
71. 🔴 pump.fun "king of the hill" reached, and how fast — *pump.fun's API no longer returns king-of-the-hill data to us*
72. 🔴 pump.fun comment (reply) count and growth — *pump.fun's API no longer returns comment counts to us*
73. 🔴 pump.fun front-page or trending placement — *no public pump.fun front-page feed*
74. 🔴 Livestream viewer count and duration — *pump.fun's API doesn't return viewer counts*
75. 🔴 Launch platform's own trending rank (LetsBonk, Bags) — *LetsBonk and Bags have no public trending API*

## E. Holders and distribution (76–93)

76. ✅ Holder count (GeckoTerminal)
77. ✅ Top-1 holder share, excluding the pool
78. ✅ Top-10 holder share, excluding the pool
79. ✅ Creator's current holding share
80. ✅ Developer holding % (GeckoTerminal)
81. ✅ Holder count growth per minute
82. ✅ Holders with more than $1k of the token
83. ✅ Gini coefficient of all holdings
84. ✅ Share of holders who bought and still hold (diamond hands)
85. ✅ Share held by wallets created in the last day
86. ✅ Share held by known exchange or market-maker wallets — *needs the `TB_CEX_WALLETS` list*
87. ✅ Number of holders also holding the last big winner (community overlap)
88. ✅ Average holder's SOL balance (rich wallets vs dust wallets)
89. ✅ Clustered holders (wallets that transfer between each other)
90. ✅ Tokens sent to many wallets for free (airdrop farming of holder count)
91. ✅ GeckoTerminal holder distribution buckets
92. ✅ Insider share at graduation (RugCheck insiders)
93. ✅ Holder churn: wallets that sold everything vs new entrants

## F. Wallet quality and behaviour (94–115)

94. ✅ Bundled launch share (non-developer buys in the launch slot)
95. ✅ Sniper share in the first 10 seconds
96. ✅ Number of early buyers
97. ✅ Largest early buyer's share
98. ✅ Concentration (Gini) of early buys
99. ✅ Equal-size buys (bot ladders)
100. ✅ Fresh-wallet share among early buyers
101. ✅ Median early-buyer wallet age
102. ✅ Same-funder share (one source funding many buyers)
103. ✅ Early buyers funded by the developer
104. ✅ Smart-money wallets (early in an earlier token that had already pumped)
105. ✅ Serial snipers (early in 5+ earlier tokens)
106. ✅ Smart-money wallet's historic win rate and size (not just a count)
107. 🔴 Smart money buying AFTER launch (later entries by proven wallets) — *only the first 100 transactions are fetched (credit budget)*
108. ✅ Known copy-trading targets buying (wallets many bots follow) — *needs the `TB_KOL_WALLETS` list*
109. ✅ Known influencer wallets buying (public KOL wallets) — *needs the `TB_KOL_WALLETS` list*
110. ✅ Early buyers that hold vs dump within 5 minutes
111. 🔴 Wallets that bought via trading bots (Photon, BullX, Axiom, Trojan) vs the pump.fun UI — *no reliable public list of bot programs and fee wallets*
112. ✅ Jito tip size on early buys (paying up to be first = conviction or a sniper)
113. ✅ Priority fees paid by buyers
114. ✅ MEV sandwich attacks on the token's buys (bots farming it = attention)
115. ✅ Funding from a centralised exchange vs a fresh chain of wallets — *needs the `TB_CEX_WALLETS` list*

## G. Developer and creator (116–133)

116. ✅ Creator's total launches on pump.fun
117. ✅ Creator's graduation rate on earlier coins
118. ✅ Creator's launches in the last 24 hours
119. ✅ Minutes since their previous launch
120. ✅ First-ever launch by this wallet
121. ✅ Reusing a name they've used before
122. ✅ Developer's initial buy share
123. ✅ Developer sold before the snapshot, and how much
124. ✅ Developer's best previous token (peak market cap)
125. 🔴 How long the developer held on previous launches — *needs full trade history of every earlier coin (credit budget)*
126. ✅ Developer wallet age and funding source
127. ✅ Developer linked to wallets of known rugs
128. ✅ Developer moved tokens to other wallets (hidden selling)
129. ✅ Developer has a pump.fun username or profile picture
130. ✅ Developer's pump.fun follower count
131. 🔴 Developer claims creator rewards immediately (cash-out behaviour) — *needs program-specific decoding of reward claims*
132. ✅ Developer is a known streamer, artist or public person
133. ✅ Same developer launching several tokens in the same theme today

## H. Contract and token security (134–145)

134. ✅ Mint authority revoked
135. ✅ Freeze authority revoked
136. ✅ Transfer fee
137. ✅ Transfer hook program
138. ✅ GeckoTerminal honeypot flag
139. ✅ RugCheck score and risk list
140. ✅ Token-2022 extensions (anything beyond plain SPL)
141. ✅ Metadata mutable or immutable
142. ✅ Update authority still held
143. ✅ Sell simulation passes (Jupiter quote for a sell)
144. ✅ Decimals and total supply (non-standard = custom contract)
145. 🔴 Program upgradeable (custom programs) — *not applicable: standard SPL tokens have no upgradeable program*

## I. Name, ticker, image, metadata (146–163)

146. ✅ Name length
147. ✅ Contains digits
148. ✅ All capitals
149. ✅ Word count
150. ✅ Image reused by a different token
151. ✅ Ticker length (3–4 letters is classic)
152. ✅ Ticker equals a common English word
153. ✅ Ticker collides with an existing well-known token
154. ✅ Name contains a trending word from X or Google Trends — *partly: via watched mega-account posts*
155. ✅ Emoji in the name
156. ✅ Non-English name (Chinese and Japanese runs happen in waves)
157. ✅ Name readability and pronounceability (meme-ability)
158. ✅ Image is AI-generated vs hand-drawn vs a photo (Haiku vision)
159. ✅ Image contains a face, an animal or text (Haiku vision)
160. ✅ Image dominant colour
161. ✅ Animated image (GIF/WebP animation)
162. ✅ Description length
163. ✅ Description tone: lore and story vs "100x moon" shilling (Haiku)

## J. Narrative and meme content (164–181)

164. ✅ Narrative category (animal, AI, political, celebrity, news, tweet, meme, brand, crypto, culture, joke)
165. ✅ Copycat of something that already ran
166. ✅ Launched off a specific social post
167. ✅ Names a real person
168. ✅ Originality score
169. ✅ Theme heat: same-word launches in the last hour
170. ✅ Theme heat: same-word pumps in the last 6 hours
171. ✅ Narrative is the hottest category this week (rotation)
172. ✅ First token on a fresh narrative vs the 20th copy
173. ✅ Narrative tied to a scheduled event (game, launch, election, holiday)
174. ✅ Narrative tied to breaking news (minutes old)
175. ✅ Narrative tied to a viral video (TikTok, YouTube Shorts)
176. ✅ Narrative tied to an AI product launch or AI meme
177. ✅ Community-takeover narrative (abandoned token revived)
178. ✅ Cult-like lore: mascot, story, recurring characters (Haiku)
179. ✅ Humour type: absurd, wholesome, edgy (Haiku)
180. ✅ Cross-chain twin: the same meme pumping on Base, BNB or Ethereum
181. 🔴 Narrative also trending on Google Trends — *Google Trends has no official API*

## K. X (Twitter) (182–199)

182. ✅ Posts containing the contract address (7 days)
183. ✅ Unique authors
184. ✅ Largest follower count among posters
185. ✅ Influencer (50k+ followers) posted
186. ✅ Verified accounts posted
187. ✅ Engagement (likes + reposts)
188. ✅ Share of posts from accounts under 30 days old
189. ✅ Twitter link type: specific post, account or none
190. ✅ Mention velocity (posts per 10 minutes, trend)
191. ✅ Posts with the ticker ($SYMBOL) and name, not just the contract address
192. ✅ Token's own X account: age, followers, follower growth
193. ✅ The linked viral post's own reach (views, likes)
194. ✅ Notable accounts replying to or quoting the linked post
195. ✅ Elon Musk or similar mega-accounts mentioning the theme
196. ✅ Bot-reply density under the posts (copy-paste shills)
197. ✅ Known paid callers posting (shill lists) — *needs the `TB_X_PAID_CALLERS` list*
198. 🔴 X Community created for the token, and its member count — *X Communities have no API*
199. ✅ Sentiment of posts (excited, sceptical, "rug")

## L. Telegram, Discord, Reddit, TikTok, YouTube and livestreams (200–214)

200. ✅ Mentions in chosen public Telegram channels
201. ✅ Number of channels that posted it
202. ✅ Minutes from launch to first channel post
203. ✅ Telegram group member count and growth (token's own group)
204. ✅ Telegram calling channel's track record (calls that pumped)
205. ✅ Discord server exists and member count
206. 🔴 Reddit posts mentioning the token or theme — *Reddit blocks or rate-limits server requests*
207. 🔴 TikTok hashtag views for the theme — *TikTok has no public API*
208. 🔴 YouTube videos on the theme in the last 24 hours — *needs a YouTube Data API key*
209. 🔴 Twitch or Kick streamer mentioning it live — *Twitch/Kick have no public search for this*
210. ✅ 4chan /biz/ mentions
211. 🔴 Farcaster mentions — *needs a Farcaster (Neynar) API key*
212. ✅ Website quality: real site vs one-page template vs none
213. ✅ Website domain age (WHOIS)
214. ✅ Linktree or Carrd-only presence

## M. Attention and listings (215–232)

215. ✅ GeckoTerminal trending, 5 minutes (rank)
216. ✅ GeckoTerminal trending, 1 hour (rank)
217. ✅ DexScreener top boosts (rank, amount)
218. ✅ DexScreener latest boosts
219. ✅ DexScreener latest profiles
220. ✅ Minutes from launch to first list appearance
221. 🔴 DexScreener trending rank (5m/1h/6h) — *DexScreener has no trending-rank API*
222. 🔴 Birdeye trending — *needs a Birdeye API key*
223. 🔴 Photon, BullX, Axiom and GMGN trending (where readable) — *private apps with no public API*
224. 🔴 Phantom wallet trending tokens — *Phantom has no public trending API*
225. ✅ Jupiter's trending or "hot" list
226. ✅ CoinGecko or CoinMarketCap new listing
227. 🔴 DexScreener "watchlist" count or reactions (rocket/fire votes) — *DexScreener reactions have no API*
228. 🔴 GMGN "smart money" tags — *GMGN blocks server requests*
229. 🔴 Featured in a pump.fun livestream compilation — *no data source*
230. 🔴 Telegram trading-bot trending lists (Trojan, BonkBot) — *Telegram bot trending lists aren't public*
231. 🔴 Listed on a points or quest platform — *no data source*
232. 🔴 Mentioned in crypto newsletters or aggregators — *no data source*

## N. Paid marketing (233–242)

233. ✅ DexScreener paid profile, and how soon after launch
234. ✅ DexScreener boosts bought
235. ✅ DexScreener ads
236. ✅ Community takeover order
237. ✅ DexScreener trending-bar ad
238. ✅ Boost size escalation (1x, 10x, 30x, 50x, 100x, 500x)
239. 🔴 Paid X promotion — *paid promotion isn't labelled in the X API*
240. 🔴 Paid Telegram calls (known paid channels) — *no list of paid Telegram channels (channel track record #204 covers part of it)*
241. ✅ Volume-bot services detected (fake volume to hit trending)
242. 🔴 Paid trending on Telegram bots — *no data source*

## O. Market conditions (243–260)

243. ✅ SOL price change, 1 hour
244. ✅ SOL price change, 24 hours
245. ✅ New pools in the last hour
246. ✅ Launchpad share of new pools
247. ✅ Recent pump rate (last 6 hours, our own data)
248. ✅ BTC price change, 1 hour and 24 hours
249. ✅ ETH price change
250. 🔴 Total pump.fun volume, last hour — *no public total-volume endpoint (launches and graduations per hour are used instead)*
251. ✅ Graduations per hour across pump.fun
252. ✅ SOL network fees and congestion (busy = mania)
253. ✅ Funding rates on SOL perpetuals (leverage appetite)
254. ✅ Stablecoin inflows to Solana
255. ✅ Fear and Greed index
256. 🔴 Big unlocks or airdrops landing (fresh money to spend) — *no free unlock/airdrop calendar API*
257. ✅ Top memecoin index (WIF, BONK, POPCAT) change, 24 hours
258. ✅ A hot token in the last 6 hours (does one big winner pull flow?)
259. ✅ US equity market open or closed
260. 🔴 Major macro events today (CPI, FOMC) — *no free macro-event calendar API*

## P. Cross-token dynamics (261–274)

261. ✅ A same-theme token is pumping right now (sympathy plays)
262. ✅ A same-theme token just rugged (contagion)
263. ✅ Number of copies launched in the last 10 minutes
264. ✅ This token is the oldest or largest of its theme ("the real one")
265. ✅ Same image or name on another chain pumping
266. ✅ Early buyers came from a token that just pumped (rotating profits)
267. ✅ Early buyers came from a token that just rugged (revenge traders)
268. ✅ Market share of its narrative category this hour
269. ✅ Competing launches at the same minute
270. 🔴 Big token's holders airdropped this token — *needs full holder lists of big tokens*
271. ✅ Paired against a memecoin instead of SOL
272. ✅ Previous token with the same ticker and how it did
273. ✅ Related-entity coins (same celebrity, same event) performance — *partly: via shared theme words*
274. ✅ Graduation queue: how many tokens graduated in the same 10 minutes

## Q. Calendar, culture and the literal stars (275–288)

Mostly noise, but cheap to test and traders believe in them, which can move flow.

275. ✅ Hour of day (UTC)
276. ✅ Weekend
277. ✅ Day of week
278. ✅ Asia, Europe or US session (who's awake)
279. ✅ US public holiday
280. ✅ Chinese and other Asian holidays (Asian retail waves)
281. ✅ Month start or end (paydays)
282. ✅ Moon phase (crypto Twitter really does post about it)
283. ✅ Mercury retrograde (same, as a sentiment marker)
284. 🔴 Major sports finals or award shows (narrative fuel) — *no free events API*
285. 🔴 Viral meme of the day (Know Your Meme, Google Trends daily) — *no free viral-meme API*
286. 🔴 Big crypto conference happening — *no free conference calendar*
287. ✅ Elon Musk posting frequency today
288. ✅ Time since the last major memecoin crash

## R. Later-stage catalysts (289–296)

289. 🔴 Centralised-exchange listing announced or rumoured — *rumours aren't machine-readable*
290. ✅ Listed on a perpetuals exchange (shorting possible)
291. 🔴 Market maker onboarded — *market makers aren't public*
292. ✅ CoinGecko or CoinMarketCap page created
293. 🔴 Partnership or collaboration announcement — *partnership announcements aren't machine-readable*
294. ✅ Team doxxed or KYC'd
295. ✅ Treasury, buyback or burn announced
296. ✅ Merch, game or app launched around the meme

## S. Manipulation signatures (297–300)

297. ✅ Wash trading (same wallets buying and selling to each other)
298. ✅ Fake volume bots (many tiny round-trip trades)
299. ✅ Coordinated pump group (many wallets buying in the same block after a call)
300. ✅ Spoofed holder count (dust sent to thousands of wallets)

---

## Counting

- ✅ built: 258 (5 of them need you to supply a wallet or account list; 2 are partial)
- 🔴 not buildable with available data: 42, each with its reason

Where each one is computed: `tradingbot/watch_analysis.py` (snapshot numbers), `tradingbot/watch_features.py`
(enrichment groups) and `tradingbot/watch_features_more.py` (the long tail; FACTORS numbers are in its comments).
