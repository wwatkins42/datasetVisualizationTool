from dateutil.parser import parse

def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True

def isDate(x):
    try:
        parse(x)
    except:
        return False
    return True
