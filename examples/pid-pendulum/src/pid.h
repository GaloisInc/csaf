/*Simple PID Controller
A PID controller is a controller with a linear control law and an associated integrator to reduce steady state error
over time. A PD controller can be created by settng the integral gain factor k_i to zero.
*/

/*Signal/System Types*/
typedef struct SignalContext {
	double signal_in;
	double signal_in_derivative;
} SignalContext;

typedef struct PidController {
	double state;
	double setpoint;
	double k_p;
	double k_d;
	double k_i;
} PidController;

/*Input Signal Methods*/
SignalContext *input_signal_init();
void input_signal_update(SignalContext *signal, double sig, double sig_d);
void input_signal_destroy(SignalContext *signal);

/* PidController Methods*/
PidController *pid_controller_init(double kp, double ki, double kd);
void pid_controller_destroy(PidController *controller);
void pid_controller_reset(PidController *controller);
void pid_controller_update(PidController *controller, SignalContext *inputs);
double pid_controller_state(PidController *controller);
double pid_controller_output(PidController *controller, SignalContext *inputs);
