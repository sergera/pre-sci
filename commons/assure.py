def type_equals(desired_types, subject, variable_name):
    message = f"type of {variable_name} must be "
    for index in range(0,len(desired_types)):
        current_type = desired_types[index]
        if isinstance(subject, current_type):
            return None
        if index == 0:
            message = message + str(current_type) + " "
        else:
            message = message + "or " + str(current_type) + " "

    message = message + f", not {str(type(subject))}"

    raise Exception(
        message
    )