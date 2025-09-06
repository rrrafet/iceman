frequency_to_multiplier = {
    "D": 365,      # Daily
    "B": 252,      # Business day
    "C": 252,      # Custom business day (treated same as B)
    
    # Weekly
    "W": 52, "W-SUN": 52, "W-MON": 52, "W-TUE": 52,
    "W-WED": 52, "W-THU": 52, "W-FRI": 52, "W-SAT": 52,
    
    # Monthly
    "M": 12,       # Month end
    "MS": 12,      # Month start
    "ME": 12,      # Alias used in some packages

    # Quarterly
    "Q": 4,        # Quarter end
    "QS": 4,       # Quarter start
    "QE": 4,
    "Q-JAN": 4, "Q-FEB": 4, "Q-MAR": 4,  # Explicit quarterly anchors
    "QS-JAN": 4, "QS-FEB": 4, "QS-MAR": 4,

    # Annual / Yearly
    "A": 1,        # Year end
    "AS": 1,       # Year start
    "Y": 1,        # Year end alias
    "YS": 1,       # Year start alias
    "A-JAN": 1, "A-FEB": 1, "A-MAR": 1,
    "AS-JAN": 1, "AS-FEB": 1, "AS-MAR": 1,
    "Y-JAN": 1, "Y-FEB": 1, "Y-MAR": 1,
    "YS-JAN": 1, "YS-FEB": 1, "YS-MAR": 1,
}