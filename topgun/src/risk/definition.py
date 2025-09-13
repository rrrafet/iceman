riskmodel = {
    "macro0": {
        "start_date": "2019-05-31",
        "end_date": "2025-08-31",
        "frequency": "B",
        "factors": [
            {
                "name": "Market",
                "ticker": ["MXWO Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Bonds",
                "ticker": ["USGG10YR Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: 0.01 * (x["USGG10YR Index"]).diff(),
            },
            {
                "name": "Dollar",
                "ticker": ["DXY Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
        ]},

    "macro1": {
        "start_date": "2019-05-31",
        "end_date": "2025-08-31",
        "frequency": "B",
        "factors": [
            {
                "name": "Market",
                "ticker": ["MXWO Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Bonds",
                "ticker": ["USGG10YR Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: 0.01 * (x["USGG10YR Index"]).diff(),
            },
            {
                "name": "Dollar",
                "ticker": ["DXY Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Commodity",
                "ticker": ["SPGSCI Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Credit",
                "ticker": ["LF98OAS Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.diff(),
            },

        ]},

    "macro2": {
        "start_date": "2019-05-31",
        "end_date": "2025-08-31",
        "frequency": "B",
        "factors": [
            {
                "name": "Market",
                "ticker": ["MXWO Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Curve",
                "ticker": ["USGG10YR Index", "USGG2YR Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: 0.01 * (x["USGG10YR Index"] - x["USGG2YR Index"]).diff(),
            },
            {
                "name": "Dollar",
                "ticker": ["DXY Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Commodity",
                "ticker": ["SPGSCI Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Credit",
                "ticker": ["LF98OAS Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.diff(),
            },
            {
                "name": "AI",
                "ticker": ["MSXXAIB Index", "SPX Index"],
                "flds": ["CHG_PCT_1D"],
                "func": lambda x: 0.01 * (x["MSXXAIB Index"] - 1.5 * x["SPX Index"]),
            },
        ],


    },
    "equity1": {
        "start_date": "2019-05-31",
        "end_date": "2025-08-31",
        "frequency": "B",
        "factors": [
            {
                "name": "Market",
                "ticker": ["MXWO Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Momentum",
                "ticker": ["MSZZMOMO Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: 0.01 * x["MSZZMOMO Index"].pct_change(),
            },
            {
                "name": "Size",
                "ticker": ["MSCBSSZU Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "Section 899",
                "ticker": ["MSST899  Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "AI",
                "ticker": ["MSXXAIB Index", "SPX Index"],
                "flds": ["CHG_PCT_1D"],
                "func": lambda x: 0.01 * (x["MSXXAIB Index"] - 1.5 * x["SPX Index"]),
            },
        ],
    },
    "equity2": {
        "start_date": "2019-05-31",
        "end_date": "2025-08-31",
        "frequency": "B",
        "factors": [
            {
                "name": "PE",
                "ticker": ["LPX50TR Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
            {
                "name": "RE",
                "ticker": ["RLSD Index"],
                "flds": ["LAST_PRICE"],
                "func": lambda x: x.pct_change(),
            },
        ],
    }
}
