# Replace the sub_string in the whole_string starting from index start
def replace(start, sub_string, whole_string):
    return whole_string[:start] + sub_string + whole_string[start + len(sub_string):]

def str_dict(dictionary):
    """
    convert a dictionary {"a": 1, "b":2} to "a:\n1\nb:\n2"
    """
    if len(dictionary) == 0:
        return ""
    else:
        str_values = ["%s:\n%s" % (key, dictionary[key]) for key in dictionary]
        return reduce(lambda x, y: "%s\n\n%s" % (x, y), str_values)
