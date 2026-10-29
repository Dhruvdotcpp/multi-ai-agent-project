_REPLACE = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-",  
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200a": " ",   
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",     
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',  
}
_TABLE = str.maketrans(_REPLACE)


def clean_text(text: str) -> str:
    return text.translate(_TABLE) if text else ""