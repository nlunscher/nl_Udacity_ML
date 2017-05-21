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

    # print predictions
    # print ages
    # print net_worths

    se = (predictions - net_worths)**2
    # print se

    se2 = []
    for e in se:
        se2.append(e[0])

    se2.sort()
    tenth = se2[-9]

    for i in range(len(predictions)):
        if se[i][0] < tenth:
            cleaned_data.append([ages[i][0], net_worths[i][0], se[i][0]])

    print cleaned_data
    
    return cleaned_data

