import redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

key = "muleguard:features:account:8139F54E0"
data = r.hgetall(key)
ttl  = r.ttl(key)
total = r.dbsize()

print(f"Total keys in Redis : {total:,}")
print(f"Key exists          : {bool(data)}")
print(f"TTL remaining       : {ttl}s")

if data:
    print(f"Number of fields    : {len(data)}")
    print("\nKey fields relevant to test:")
    watch = ("sender_receiver_pair_txn_count", "sender_avg_amount",
             "currency_pair_rarity", "fan_out_24h", "txn_count_1h",
             "day_of_month", "minute_of_hour")
    for k in watch:
        print(f"  {k:40s} = {data.get(k, 'MISSING')}")
else:
    print("Account 8139F54E0 NOT FOUND in Redis")
    # Sample a few keys to confirm Redis is working
    sample = list(r.scan_iter("muleguard:features:account:*", count=5))[:3]
    print(f"Sample keys: {sample}")
