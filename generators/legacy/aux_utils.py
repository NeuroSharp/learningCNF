
def is_number(s):
    try:
        int(s)
    except ValueError:
        return False
    return True

def sign(x):
    return x and (1, -1)[x < 0]
