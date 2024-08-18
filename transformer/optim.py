def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.

    Linear Part
        + The term step * warmup^(-1.5) is linear with respect to step.
        + As step increases, this part of the expression grows linearly.
    Non-Linear Part
        + The term step^(-0.5) is non-linear with respect to step.
        + As step increases, this part of the expression decreases at a rate proportional 
            to the square root of step.
    Overall Structure
        + The min() function is used to select the smaller value between the two terms:
            step^(-0.5) and step * warmup^(-1.5).
        + Initially, when step is small, the linear term (i.e., step * warmup^(-1.5)) 
            might dominate, but as step increases, the non-linear term (step^(-0.5)) 
            may take over.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )