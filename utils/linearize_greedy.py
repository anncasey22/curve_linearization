def linearize_curve(points, epsilon=0.01):
    """"
    greedy method that gives the piecewise linear approx of a curve
    - extend each line segment until error exceeds threshold
    - check max deviation of points in between line 
    """""
    result = [points[0]] #start with the first point
    start = 0
    for i in range(2, len(points)):
        x1, y1 = points[start]
        x2, y2 = points[i]

        dx, dy = x2 - x1, y2 - y1
    #create a line between the start point and current point
        if dx == 0:
            continue
        slope = dy/ dx
        intercept = y1 - slope * x1

    #check error for all points between the start and i
        max_error = 0
        for j in range(start + 1, i):
            xj, yj = points[j]
            y_prev = slope * xj + intercept
            error = abs(y_prev - yj)
            max_error = max(max_error,error)

    #check if the error is too much
        if max_error > epsilon:
            result.append(points[i - 1])
            start = i - 1

        result.append(points[i-1]) #final point
        return result






