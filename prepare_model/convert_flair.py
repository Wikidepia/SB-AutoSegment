import ujson

with open("combined.txt") as f:
    for line in f:
        try:
            data = ujson.loads(line)
            data = data[:2000]
            if all(x[1] == "CONTENT" for x in data):
                continue
            for ts in data:
                text = ts[0]
                if text.startswith("["):
                    continue
                text = text.strip()
                if text == "":
                    continue
                text = text.split()[0]
                print(text, ts[1])
            print()
        except Exception as e:
            pass
