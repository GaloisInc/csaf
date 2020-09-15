#include <stdlib.h>
#include <string.h>
#include "pid.h"


SignalContext *input_signal_init(){
    /* Signal Initializer */
	SignalContext* signals;
	signals = malloc(sizeof(SignalContext));
	memset(signals, 0, sizeof(SignalContext));
	return (SignalContext *)signals;
}

void input_signal_destroy(SignalContext *signal){
    /*Signal Context Destructor*/
    if (signal) free(signal);
}

void input_signal_update(SignalContext *signal, double sig, double sig_d){
    /* Signal Setter */
	signal->signal_in = sig;
	signal->signal_in_derivative = sig_d;
}

PidController *pid_controller_init(double kp, double ki, double kd){
	/* PID Controller Initialization */
	PidController *controller;
	controller = (PidController *)malloc(sizeof(PidController));
	memset((void *)controller, 0, sizeof(PidController));
	pid_controller_reset(controller);
	controller->k_p = kp;
	controller->k_i = ki;
	controller->k_d = kd;
	return controller;
}

void pid_controller_destroy(PidController *controller){
    /*PID destructor*/
    if (controller) free(controller);
}

void pid_controller_reset(PidController *controller){
	/* PID Controller State Reset -- zero out the integrator */
	controller->state = 0.0;
}

void pid_controller_set_state(PidController *controller, double state){
	/* PID Controller State Set */
	controller->state = state;
}

void pid_controller_update(PidController *controller, SignalContext *inputs){
	/* PID Controller State Update Function */
	double error_signal;
	error_signal = controller->setpoint - inputs->signal_in;
	controller->state += error_signal;
}

double pid_controller_state(PidController *controller){
    /*get the PID controller state*/
    return controller->state;
}

double pid_controller_output(PidController *controller, SignalContext *inputs){
	/* PID Controller Output Function */
	double error_signal, error_signal_d, control_law;
	error_signal = controller->setpoint - inputs->signal_in;
	error_signal_d = - inputs->signal_in_derivative;
	control_law = controller->k_p * error_signal + 
		controller->k_d * error_signal_d + 
		controller->k_i * controller->state;
	return control_law;
}

