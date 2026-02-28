import requests
import zipfile
import polars as pl
import os
import io
import argparse
from dotenv import load_dotenv
from huggingface_hub import list_repo_files, upload_file, login

load_dotenv()

repo_id = "SleepPenguin/bfuus"
process_interval = ["1m", "5m", "15m", "30m", "1h"]

login(token=os.getenv("HF_TOKEN"))
files = list_repo_files(repo_id=repo_id, repo_type="dataset")
# print(files)


def get_all_urls(url_file):
    with open(url_file, "r") as f:
        urls = f.read().splitlines()
    return urls


def parse_url(url: str):
    if "aggTrades" not in url:
        raise ValueError("URL does not contain 'aggTrades'")
    if "futures" not in url:
        raise ValueError("URL does not contain 'futures'")
    if "monthly" not in url:
        raise ValueError("URL does not contain 'monthly'")
    file_name = url.split("/")[-1]
    symbol = file_name.split("-")[0]
    month_str = "-".join(file_name.split("-")[-2:]).replace(".zip", "")
    res = {
        "url": url,
        "file_name": file_name,
        "symbol": symbol,
        "month_str": month_str,
    }
    if not symbol.endswith("USDT"):
        raise ValueError(f"Parse failed, Symbol {symbol} does not end with 'USDT'")
    print(f"Parsed URL: {res}")
    return res


def download_zip_file(url):
    print(f"Downloading from URL: {url}")
    response = requests.get(url)
    response.raise_for_status()
    print(f"Downloaded {len(response.content)} bytes from {url}")
    return response.content


def content_to_lf(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            lf = pl.scan_csv(
                f,
                has_header=False,
                schema={
                    "agg_trade_id": pl.Int64,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64,
                    "transact_time": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                },
                ignore_errors=True,
            ).drop_nulls(subset=["agg_trade_id"])
    return lf


def join_to_interval(lf: pl.LazyFrame, interval):
    lf = lf.with_columns(
        pl.col("transact_time").cast(pl.Datetime(time_unit="ms")).alias("datetime")
    )
    num_trades = pl.col("last_trade_id").last() - pl.col("first_trade_id").first() + 1
    is_taker = 1 - pl.col("is_buyer_maker").cast(pl.Int8)
    taker_buy_volume = (pl.col("quantity") * is_taker).sum()
    taker_buy_quote_volume = (pl.col("quantity") * pl.col("price") * is_taker).sum()
    lf = lf.group_by_dynamic("datetime", every=interval).agg(
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("quantity").sum().alias("volume"),
        (pl.col("quantity") * pl.col("price")).sum().alias("quote_volume"),
        num_trades.alias("num_trades"),
        taker_buy_volume.alias("taker_buy_volume"),
        taker_buy_quote_volume.alias("taker_buy_quote_volume"),
    )
    lf = lf.with_columns(pl.lit(interval).alias("interval").cast(pl.Categorical))
    return lf


def get_kline_out_path(symbol, month_str, interval):
    return f"klines/interval={interval}/symbol={symbol}/month={month_str}/data.parquet"


def get_agg_trade_out_path(symbol, month_str):
    return f"agg_trades/symbol={symbol}/month={month_str}/data.parquet"


def upload_to_hf(out_path: str, lf: pl.LazyFrame, symbol: str, month_str: str, skip_exist=True):
    if skip_exist and out_path in files:
        print(f"File {out_path} already exists in Hugging Face Hub. Skipping upload.")
        return
    lf = lf.with_columns(
        pl.lit(symbol).alias("symbol").cast(pl.Categorical),
        pl.lit(month_str).alias("month").cast(pl.Categorical),
    )
    # 保存到huggingface hub
    lf.sink_parquet(out_path, compression="zstd", compression_level=3, mkdir=True)
    upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Uploaded {out_path}",
    )
    print(f"Uploaded success {out_path} to Hugging Face Hub under repo {repo_id}")
    # 删除本地文件
    os.remove(out_path)
    print(f"Removed local file: {out_path}")


def process_one_url(url):
    parse_res = parse_url(url)
    out_path = get_agg_trade_out_path(parse_res["symbol"], parse_res["month_str"])
    if out_path in files:
        print(f"{url} already exists in Hugging Face Hub. Skipping download and processing.")
        return
    content = download_zip_file(url)
    lf = content_to_lf(content)
    upload_to_hf(out_path, lf, parse_res["symbol"], parse_res["month_str"])
    for interval in process_interval:
        print(f"Processing interval: {interval}")
        interval_lf = join_to_interval(lf, interval)
        out_path = get_kline_out_path(
            parse_res["symbol"], parse_res["month_str"], interval
        )
        upload_to_hf(out_path, interval_lf, parse_res["symbol"], parse_res["month_str"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", help="Path to the file containing URLs to process")
    args = parser.parse_args()
    urls = get_all_urls(args.url_file)
    for url in urls:
        try:
            process_one_url(url)
        except Exception as e:
            print(f"Error processing URL: {url}. Error: {e}")
            with open("error.log", "a") as log_file:
                log_file.write(f"{url}\nError: {e}\n")
