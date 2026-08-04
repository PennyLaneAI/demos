import string

bad_chars=b""  # B Skinn 2021-12-11
for i in range(128,256):
    bad_chars+=chr(i).encode()  # B Skinn 2021-12-11
table_from=string.punctuation+string.ascii_uppercase
table_to=' '*len(string.punctuation)+string.ascii_lowercase
trans_table=bytes.maketrans(table_from.encode(), table_to.encode())  # B Skinn 2021-12-11


def asciionly(s):
    return s.encode().translate(None, bad_chars).decode(errors='replace')  # B Skinn 2021-12-11

# remove non-ASCII characters from strings
def asciidammit(s):
    if type(s) is str:
        return asciionly(s)
    elif type(s) is unicode:
        return asciionly(s.encode('ascii', 'ignore'))
    else:
        return asciidammit(unicode(s))

def validate_string(s):
    try:
        if len(s)>0:
            return True
        else:
            return False
    except:
        return False

def full_process(s):
    s = asciidammit(s)
    # B Skinn 2021-12-11
    return s.encode().translate(trans_table, bad_chars).decode(errors='replace').strip()
