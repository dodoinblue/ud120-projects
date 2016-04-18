#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error = []
    import math
    for i in range(len(ages)):
        error.append(math.fabs(predictions[i] - net_worths[i]))

    threshold = sorted(error)[-9]

    zipped = zip(ages, net_worths, error)

    for data in zipped:
        if data[2] < threshold:
            cleaned_data.append(data)
    
    print 'number of data left: ', len(cleaned_data)
    
    return cleaned_data

