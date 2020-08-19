def p_cntrl(kp, e):
    return pid_cntrl(kp=kp, kd=0, ki=0, e=e, ed=0, ei=0)

def pd_cntrl(kp, kd, e, ed):
    return pid_cntrl(kp=kp, kd=kd, ki=0, e=e, ed=ed, ei=0)

def pi_cntrl(kp, ki, e, ei):
    return pid_cntrl(kp=kp, kd=0, ki=ki, e=e, ed=0, ei=ei)

def pid_cntrl(kp, kd, ki, e, ed, ei):
    return e*kp + ed*kd + ei*ki
