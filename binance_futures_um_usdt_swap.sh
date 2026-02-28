#!/usr/bin/env bash
set -euo pipefail

# 获取所有aggtrades链接
python gen_url.py --pattern "data/futures/um/monthly/aggTrades/SYMBOL/" --symbol-glob "*USDT"
# 下载并处理数据，上传到Hugging Face Hub
python spider_um.py --url-file "data_futures_um_monthly_aggTrades_SYMBOL_.txt"
# 打包日志
# zip files
DATE=$(date +%Y%m%d)
mkdir -p logs
ZIP_PATH=logs/${DATE}.zip
zip -j ${ZIP_PATH} data_futures_um_monthly_aggTrades_SYMBOL_.txt error.log