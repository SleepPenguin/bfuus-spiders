"""
Example usage:
python gen_url.py --pattern "data/spot/daily/klines/SYMBOL/1m/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/daily/klines/SYMBOL/1m/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/monthly/fundingRate/SYMBOL/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/daily/metrics/SYMBOL/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/daily/indexPriceKlines/SYMBOL/1m/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/monthly/markPriceKlines/SYMBOL/1m/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/daily/premiumIndexKlines/SYMBOL/1m/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/monthly/aggTrades/SYMBOL/" --symbol-glob "*USDT"
python gen_url.py --pattern "data/futures/um/monthly/klines/SYMBOL/1m/" --symbol-glob "*USDT"
"""


import asyncio
import argparse
import aiohttp
import re
import xml.etree.ElementTree as ElementTree
from typing import List
import fnmatch
import aiofiles

BASE_URL = "https://data.binance.vision"


def build_url_file_name(pattern: str, symbol_glob: str) -> str:
    safe_pattern = pattern.replace("/", "_")
    return f"{safe_pattern}.txt"


def parse_s3_listing(xml_text: str, prefix: str):
    root = ElementTree.fromstring(xml_text.encode("utf-8"))
    namespace = {"s3": root.tag.split("}")[0].strip("{")}
    entries: List[str] = []
    for element in root.findall(".//s3:CommonPrefixes/s3:Prefix", namespaces=namespace):
        if not element.text:
            continue
        name = element.text[len(prefix) :].strip("/")
        if name:
            entries.append(name)
    for element in root.findall(".//s3:Contents/s3:Key", namespaces=namespace):
        if not element.text:
            continue
        key = element.text
        if not key.endswith(".zip"):
            continue
        name = key[len(prefix) :]
        if name:
            entries.append(name)
    is_truncated = (
        root.findtext(".//s3:IsTruncated", default="false", namespaces=namespace)
        .strip()
        .lower()
        == "true"
    )
    next_marker = root.findtext(".//s3:NextMarker", namespaces=namespace)
    last_key = root.findtext(".//s3:Contents/s3:Key[last()]", namespaces=namespace)
    return entries, is_truncated, next_marker, last_key


class BinanceDataURLSpider:
    def __init__(self, max_retries=3, req_timeout=10, max_concurrency=5):
        self.max_retries = max_retries
        self.req_timeout = req_timeout
        self.failed_file = "failed.txt"
        self.max_concurrency = max_concurrency

    async def log_failed(self, url: str):
        async with aiofiles.open(self.failed_file, "a", encoding="utf-8") as f:
            await f.write(f"{url}\n")

    async def fetch_text(self, session: aiohttp.ClientSession, url: str) -> str:
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, timeout=self.req_timeout) as resp:
                    resp.raise_for_status()
                    return await resp.text()
            except Exception as e:
                print(f"Error fetching {url}: {e}. Retry times: {attempt + 1}")
                await asyncio.sleep(2**attempt)
        await self.log_failed(url)
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} retries")

    async def get_bucket_url(self, session: aiohttp.ClientSession, prefix: str) -> str:
        listing_url = f"{BASE_URL}/?prefix={prefix}"
        text = await self.fetch_text(session, listing_url)
        match = re.search(r"var BUCKET_URL = '(.*?)';", text)
        return match.group(1)

    async def list_prefix(self, session: aiohttp.ClientSession, prefix: str):
        bucket_url = await self.get_bucket_url(session, prefix)
        entries = []
        continuation = None
        while True:
            params = f"delimiter=/&prefix={prefix}&max-keys=1000"
            if continuation:
                params += f"&marker={continuation}"
            request_url = f"{bucket_url}?{params}"
            text = await self.fetch_text(session, request_url)
            page_entries, is_truncated, next_marker, last_key = parse_s3_listing(
                text, prefix
            )
            entries.extend(page_entries)
            if not is_truncated:
                break
            continuation = next_marker or last_key
            if not continuation:
                break
        return entries

    async def generate_urls(self, pattern: str, symbol_glob: str):
        # async with aiohttp.ClientSession() as session:
        async with aiohttp.ClientSession(proxy='http://127.0.0.1:7897') as session:
            endpoint = pattern.split("SYMBOL")[0]
            all_symbols = await self.list_prefix(session, endpoint)
            symbols = [s for s in all_symbols if fnmatch.fnmatch(s, symbol_glob)]
            print(f"Matched symbols: {len(symbols)}, all symbols: {len(all_symbols)}")
            prefixes = [pattern.replace("SYMBOL", symbol) for symbol in symbols]
            sem = asyncio.Semaphore(self.max_concurrency)

            # print(urls)
            async def run_one_task(pf: str):
                async with sem:
                    print(f"Processing prefix: {pf}")
                    entries = await self.list_prefix(session, prefix=pf)
                    urls = [f"{BASE_URL}/{pf}{et}" for et in entries]
                    return urls

            tasks = [run_one_task(pf) for pf in prefixes]
            results = await asyncio.gather(*tasks)
            all_urls = [url for sublist in results for url in sublist]
            output_file = build_url_file_name(pattern, symbol_glob)
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                for url in all_urls:
                    await f.write(f"{url}\n")
            print(f"Generated {len(all_urls)} URLs, saved to {output_file}.")


async def main(args):
    spider = BinanceDataURLSpider()
    await spider.generate_urls(args.pattern, args.symbol_glob)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern")
    parser.add_argument("--symbol-glob")
    parse_args = parser.parse_args()
    print(
        f"Generating URLs for pattern: {parse_args.pattern}, symbol glob: {parse_args.symbol_glob}"
    )
    asyncio.run(main(parse_args))
