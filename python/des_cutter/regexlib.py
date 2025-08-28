import string
import re

# Regular expression to catch ${variable} format in string
REGEX = r"\${(.*?)\}"


def get_var_names(d):
    """
    Get all matches and store them using regexp for variables with the format:
    ${variable}. This will be updated later.
    """

    # Regular expression to catch ${variable} format in string
    all_matches = []
    for key, val in d.items():
        # If a dictionary... go inside -- recursive
        if isinstance(val, dict):
            matches = get_var_names(val)
            all_matches.extend(x for x in matches if x not in all_matches)
        else:
            matches = re.findall(REGEX, str(val), re.MULTILINE | re.DOTALL)
            all_matches.extend(x for x in matches if x not in all_matches)
    return all_matches


def get_var_dict(d, all_matches):
    """
    Build a dictionary with all the variables that were matched with
    the values populated
    --------
    d: dictionary
       The input dictionary
    all_matches: List
       The input list with all variables that have a match
    Returns
    -------
    vars_dict: dictionary with values with matches
    """
    vars_dict = {}
    for match in all_matches:
        vars_dict[match] = d[match]
    return vars_dict


def update_var_dict(d, var_dict):
    """
    Updates the matches in dictionary with the new values
    """
    # Loop over the items
    for key, val in d.items():
        if isinstance(val, dict):
            val = update_var_dict(val, var_dict)
            d[key] = val
        else:
            d[key] = update_var_value(val, var_dict)
    return d


def update_var_value(val, var_dict):

    """
    For a given value (string) in a dictionary replaces ${variable}
    by the value stored in var_dict
    """

    s = str(val)
    matches = re.findall(REGEX, s, re.MULTILINE | re.DOTALL)
    if len(list(matches)) > 0:
        # Build the dictionary with kwargs
        kw = {}
        for match in matches:
            kw[match] = var_dict[match]
        template = string.Template(s)
        newval = template.safe_substitute(**kw)
        try:
            newval = eval(newval)
        except Exception:
            pass
    else:
        newval = val
    return newval


def replace_variables_in_dict(d):
    """
    Replace all variables in the dictionary with the format ${variable} by the
    value defined as variable:value in the same dictionary itself. The
    functions need to be called one at a time as these are recursive functions.
    Probably there is a simpler way to do this.... :(
    """
    var_names = get_var_names(d)
    var_dict = get_var_dict(d, var_names)
    new_dict = update_var_dict(d, var_dict)
    return new_dict
