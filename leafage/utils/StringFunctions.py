# Replace the sub_string in the whole_string starting from index start
def replace(start, sub_string, whole_string):
    return whole_string[:start] + sub_string + whole_string[start + len(sub_string):]


def str_dict_new_line(dictionary):
    """
    convert a dictionary {"a": 1, "b":2} to "a:\n1\nb:\n2"
    """
    if len(dictionary) == 0:
        return ""
    else:
        str_values = ["%s:\n%s" % (key, dictionary[key]) for key in dictionary]
        return reduce(lambda x, y: "%s\n\n%s" % (x, y), str_values)


def str_dict_snake(dictionary):
    """
    Convert {"a":1,"b":2} to a_1_b_2
    """
    if dictionary is None or len(dictionary) == 0:
        return ""
    else:
        return reduce(lambda i, j: "%s_%s" % (i, j), ["%s_%s" % (x, dictionary[x]) for x in dictionary.keys()])