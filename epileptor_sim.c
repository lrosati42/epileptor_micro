/************************************************************
  epileptor_sim.c

  Example translation of the Python-based two-population 
  neuronal model into ANSI C.

  Compile: 
     gcc -std=c99 -o epileptor_sim epileptor_sim.c -lm
  Run:
     ./epileptor_sim

************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------------
   Helper: Uniform random in [min_val, max_val].
   NOTE: For higher-quality PRNG, replace rand() with a better generator.
   -i----------------------------------------------------------------- */
double uniform_rand(double min_val, double max_val)
{
    return min_val + (max_val - min_val) * ((double)rand() / (double)RAND_MAX);
}

double linear_sigmoid(double x) {
    if (x < -2.0)
        return 0.0;
    else if (x > 2.0)
        return 1.0;
    else
        return 0.5 + 0.25 * x;  // linear interpolation: S(-2)=0, S(2)=1.
}

double linear_tanh(double x) {
    if (x <= -2.0)
        return -1.0;
    else if (x >= 2.0)
        return 1.0;
    else
        return 0.5 * x;  // approximates tanh(x) linearly on [-2, 2]
}

double linear_cosh(double x) {
    double ax = (x < 0 ? -x : x);
    if (ax >= 2.0)
        return 3.7622;  // approximate cosh(2) ≈ 3.7622
    else {
        // Linear interpolation between (0,1) and (2,3.7622)
        double slope = (3.7622 - 1.0) / 2.0;  // ≈ 1.3811
        return 1.0 + slope * ax;
    }
}

/* ======================= SLOW SYNAPSE ======================= */
typedef struct {
    double gmax;   
    double Esyn;  
    double a;     
    double b;     
    double Cmax;  
    double threshold;
    double Cdur;
    double deadtime;
    double lr;     /* last release time (not actively used here) */

    /* gating variables */
    double m, m0, tau_r, m1; 
    double Kd;
    double K3; 
    double K4; 
    //   double n;   	/* used to be exponent for s^n in I calculation */  
    //   		/* now n is hardwire to 4 see "comment.1"	*/
    double r;   /* gating var */
    double s;   /* gating var */
    double Vt;  /* presyn threshold param */
} slow_synapse;

/* Returns synaptic current for slow synapse. Mirrors Python eq() */
double slow_synapse_eq(slow_synapse *syn, double Vpre, double Vpost, double dt, double t)
{
    double I    = 0.0;
    double Cmax = 1.0; 
    double Kp   = 5.0;

    /* Piece-wise linear transmitter function */
    double sig_arg = (Vpre - syn->Vt) / Kp;
    double C = Cmax * linear_sigmoid(sig_arg);  

    /* Update r, s */
    syn->r += (syn->a*C*(1.0 - syn->r) - syn->b*syn->r) * dt;
    syn->s += (syn->K3*syn->r - syn->K4*syn->s) * dt;

    /* previosly Current:  I = -gmax * (s^n) * (Vpost - Esyn)/(Kd + s^n) */
    // comment.1  now n is hardwired to 4    was:  double s_pow_n = pow(syn->s, syn->n);
    double s_pow_n = syn->s * syn->s * syn->s * syn->s;
    I = -syn->gmax * s_pow_n * (Vpost - syn->Esyn) / (syn->Kd + s_pow_n);

    return I;
}

/* ======================= FAST SYNAPSE ======================= */
typedef struct {
    double gmax;
    double Esyn;
    double a;
    double b;
    double Cmax;
    double threshold;
    double Cdur;
    double deadtime;
    double lr;  /* last release (unused here) */

    double m, m0, tau_r, m1;
    double Vt;
} fast_synapse;

/* Returns synaptic current for fast synapse. Mirrors Python eq() */
double fast_synapse_eq(fast_synapse *syn, double Vpre, double Vpost, double dt, double t)
{
    double I    = 0.0;
    double Cmax = 1.0;
    double Kp   = 5.0;

    double sig_arg = (Vpre - syn->Vt) / Kp;
    double C = Cmax * linear_sigmoid(sig_arg);

    /* Update m */
    syn->m += (syn->a*C*(1.0 - syn->m) - syn->b*syn->m)*dt;

    /* Current: -gmax*m*(Vpost - Esyn) */
    I = -syn->gmax * syn->m * (Vpost - syn->Esyn);

    return I;
}

/* Forward reference to pop1n so pop2n can point to it. */
typedef struct pop1n pop1n;

/* ======================= POP2N =======================
   (Morris-Lecar style neuron)
   We define pop2n before pop1n so that pop1n_euler can 
   safely reference pop2n->x2, etc.
   ===================================================== */
typedef struct pop2n {
    /* Model parameters */
    double aa, tau, I2, z0;

    /* State variables */
    double x2;
    double y2;

    /* Morris-Lecar parameters */
    double V1, V2, V3, V4, phi;
    double gCa_bar, gK_bar, gL_bar;
    double ECa, EK, EL;
    double Cm;

    /* Synapses */
    fast_synapse syn_x2x2;
    fast_synapse syn_x1x2;
    slow_synapse syn_x2x2_slow;

    /* Coupling strengths */
    double CpES; /* gap-junction coupling */
    double CpCS; /* chemical coupling */

    /* References to neurons in each population */
    pop1n **pop1in;   /* all pop1n neurons that project to me */
    int pop1in_count;

    struct pop2n **pop2in;      /* other pop2n neurons */
    int pop2in_count;

    struct pop2n **pop2in_gj;   /* pop2n neurons for gap junction */
    int pop2in_gj_count;

    double c2;
    double noise;
} pop2n;

/* ======================= POP1N =======================
   (Hindmarsh-Rose style neuron)
   We can now safely define pop1n, referencing pop2n fully.
   ===================================================== */
struct pop1n {
    /* Epileptor parameters */
    double a, b, c, d, m, s, x0, z0, r, I1;

    /* State variables */
    double x1, y1, z;

    /* Synapses inside pop1n struct */
    fast_synapse syn_x1x1;
    fast_synapse syn_x2x1;
    slow_synapse syn_x2x1_slow;

    /* Coupling strengths */
    double CpES;  /* gap-junction coupling */
    double CpCS;  /* chemical coupling */

    /* References to other neurons */
    pop1n **pop1in_gj; 
    int pop1in_gj_count;

    pop1n **pop1in;    
    int pop1in_count;

    pop2n **pop2in;    
    int pop2in_count;

    /* Noise parameters */
    double noise;
    double noise3;
};

/* ===========================================================
   pop2n_euler: Euler step for one pop2n neuron
   =========================================================== */
void pop2n_euler(pop2n *neuron,
                 double dt,
                 double t,
                 double x1bar, double x2bar, double zbar)
{
    /* Compute average x2 from pop2in */
    double x2in = 0.0;
    if (neuron->pop2in_count > 0) {
        for(int i=0; i< neuron->pop2in_count; i++){
            x2in += neuron->pop2in[i]->x2;
        }
        x2in /= neuron->pop2in_count;
    }

    /* Compute average x1 from pop1in */
    double x1in = 0.0;
    if (neuron->pop1in_count > 0) {
        for(int i=0; i< neuron->pop1in_count; i++){
            x1in += neuron->pop1in[i]->x1;
        }
        x1in /= neuron->pop1in_count;
    }

    /* Compute average x2 from gap-junction group pop2in_gj */
    double x2in_gj = 0.0;
    if (neuron->pop2in_gj_count > 0) {
        for(int i=0; i< neuron->pop2in_gj_count; i++){
            x2in_gj += neuron->pop2in_gj[i]->x2;
        }
        x2in_gj /= neuron->pop2in_gj_count;
    }

    /* Evaluate synaptic currents (scaled by 50.0, as in Python code) */
    double Isyn_x2x2      = fast_synapse_eq(&(neuron->syn_x2x2),
                                            x2in*50.0, neuron->x2*50.0, dt, t*dt);
    double Isyn_x2x2_slow = slow_synapse_eq(&(neuron->syn_x2x2_slow),
                                            x2in*50.0, neuron->x2*50.0, dt, t*dt);
    double Isyn_x1x2      = fast_synapse_eq(&(neuron->syn_x1x2),
                                            x1in*50.0, neuron->x2*50.0, dt, t*dt);

    /* Gap junction current: CpES*(x2in_gj - x2)*50 */
    double Igap_x2x2 = neuron->CpES * (x2in_gj - neuron->x2) * 50.0;

    /* Convert x2 => membrane voltage V */
    double V = neuron->x2 * 20.0;
    double n = neuron->y2;

    /* Build total input current (similar to the Python code) */
    double I_in = neuron->I2*50.0
                  - neuron->c2*(zbar - 3.0)*50.0
                  + neuron->CpCS*Isyn_x2x2
                  + Igap_x2x2
                  + Isyn_x1x2
                  + Isyn_x2x2_slow;  /* slow syn */

    /* Morris-Lecar sub-expressions */
	// Piecewise linear approximation
	double m_inf = 0.5 * (1.0 + linear_tanh((V - neuron->V1) / neuron->V2));
	double n_inf = 0.5 * (1.0 + linear_tanh((V - neuron->V3) / neuron->V4));
    	double tau_n = 1.0 / linear_cosh((V - neuron->V3) / (2.0 * neuron->V4));

    double dn  = neuron->phi * (n_inf - n) / tau_n;
    double dVm = (I_in 
                  - neuron->gCa_bar*m_inf*(V - neuron->ECa)
                  - neuron->gK_bar*n*(V - neuron->EK)
                  - neuron->gL_bar*(V - neuron->EL)) 
                  / neuron->Cm;

    /* Add noise to dVm (scaled by 50.0, as in the Python code) */
    double noise_val = uniform_rand(-neuron->noise, neuron->noise)*50.0;
    dVm += noise_val / neuron->Cm;

    /* Euler updates */
    double n_next = n + dn*dt;
    double V_next = V + dVm*dt;

    neuron->y2 = n_next;
    neuron->x2 = V_next / 20.0;
}

/* ===========================================================
   pop1n_euler: Euler step for one pop1n neuron
   =========================================================== */
void pop1n_euler(pop1n *neuron,
                 double dt,
                 double t,
                 double x1bar, double x2bar, double zbar)
{
    double a = neuron->a, b = neuron->b, c = neuron->c, d = neuron->d;
    double m_ = neuron->m, s_ = neuron->s, x0 = neuron->x0, r_ = neuron->r;
    double I1 = neuron->I1;

    /* Average x2 from pop2in */
    double x2in = 0.0;
    if(neuron->pop2in_count > 0) {
        for(int i=0; i< neuron->pop2in_count; i++){
            x2in += neuron->pop2in[i]->x2;
        }
        x2in /= neuron->pop2in_count;
    }

    /* Average x1 from pop1in */
    double x1in = 0.0;
    if(neuron->pop1in_count > 0) {
        for(int i=0; i< neuron->pop1in_count; i++){
            x1in += neuron->pop1in[i]->x1;
        }
        x1in /= neuron->pop1in_count;
    }

    /* Average x1 from gap-junction group */
    double x1in_gj = 0.0;
    if(neuron->pop1in_gj_count > 0) {
        for(int i=0; i< neuron->pop1in_gj_count; i++){
            x1in_gj += neuron->pop1in_gj[i]->x1;
        }
        x1in_gj /= neuron->pop1in_gj_count;
    }

    /* Synaptic currents: scale presyn/postsyn by 50 (like Python) */
    double Isyn_x2x1      = fast_synapse_eq(&(neuron->syn_x2x1),
                                            x2in*50.0, neuron->x1*50.0, dt, t*dt);
    double Isyn_x2x1_slow = slow_synapse_eq(&(neuron->syn_x2x1_slow),
                                            x2in*50.0, neuron->x1*50.0, dt, t*dt);
    double Isyn_x1x1      = fast_synapse_eq(&(neuron->syn_x1x1),
                                            x1in*50.0, neuron->x1*50.0, dt, t*dt);

    /* Gap junction current */
    double Igj_x1x1 = neuron->CpES * (x1in_gj - neuron->x1);

    /* Noise terms */
    double noise_term   = uniform_rand(-neuron->noise,  neuron->noise);
    double noise3_term  = uniform_rand(-neuron->noise3, neuron->noise3);

    /* Euler update for x1, y1, z  */
    double x1_cube = neuron->x1 * neuron->x1 * neuron->x1;
    double x1_square = neuron->x1 * neuron->x1;
    double x1_next = neuron->x1 
       + ( neuron->y1 
           - a * x1_cube
           + b * x1_square
           - neuron->z
           + I1
       + neuron->CpCS*(Isyn_x1x1/50.0)
       + Igj_x1x1
       + (Isyn_x2x1/50.0)
       + (Isyn_x2x1_slow/50.0)
       + noise_term
     )*dt;
	
    double y1_next = neuron->y1 + ( c - d * x1_square - neuron->y1 ) * dt;

    double z_next = neuron->z 
       + ( r_*( s_*( neuron->x1 + x2bar - x0 ) - zbar ) + noise3_term )*dt;

    /* Store back */
    neuron->x1 = x1_next;
    neuron->y1 = y1_next;
    neuron->z  = z_next;
}

/* ==================== Connection Helpers ==================== */

void connect_pop1n_gap(pop1n *target, pop1n **group, int count)
{
    target->pop1in_gj = group;
    target->pop1in_gj_count = count;
}

void connect_pop1n_syn_pop1n(pop1n *target, pop1n **group, int count)
{
    target->pop1in = group;
    target->pop1in_count = count;
}

void connect_pop1n_syn_pop2n(pop1n *target, pop2n **group, int count)
{
    target->pop2in = group;
    target->pop2in_count = count;
}

void connect_pop2n_gap(pop2n *target, pop2n **group, int count)
{
    target->pop2in_gj = group;
    target->pop2in_gj_count = count;
}

void connect_pop2n_syn_pop2n(pop2n *target, pop2n **group, int count)
{
    target->pop2in = group;
    target->pop2in_count = count;
}

void connect_pop2n_syn_pop1n(pop2n *target, pop1n **group, int count)
{
    target->pop1in = group;
    target->pop1in_count = count;
}

/* ==================== MAIN EXAMPLE ==================== */
int main(void)
{
    srand((unsigned int)time(NULL)); // seed

    // Misura il tempo di inizio
    clock_t start_time = clock();

    double dt   = 0.05;
    double T    = 1000.0;  // total simulation time in ms
    int steps   = (int)(T/dt);

    /* We'll have N1 pop1n neurons and N2 pop2n neurons */
    int N1 = 40;
    int N2 = 40;

    /* Allocate pop2n first (to avoid incomplete type issues) */
    pop2n *population2 = (pop2n *)malloc(N2 * sizeof(pop2n));
    for(int i=0; i<N2; i++) {
        population2[i].aa  = 8.0; 
        population2[i].tau = 20.0;
        population2[i].I2  = 0.8;
        population2[i].z0  = 0.0;

        population2[i].x2  = uniform_rand(-1.25, 1.0);
        population2[i].y2  = uniform_rand(0.0, 1.0);
        population2[i].c2  = 0.3;
        population2[i].noise = 0.1;

        population2[i].CpES = 0.8;
        population2[i].CpCS = 1.0;

        /* Morris-Lecar */
        population2[i].V1 = -1.2;
        population2[i].V2 = 18.0;
        population2[i].V3 = 12.0;
        population2[i].V4 = 17.4;
        population2[i].phi= 0.067;
        population2[i].gCa_bar = 4.0;
        population2[i].gK_bar  = 8.0;
        population2[i].gL_bar  = 2.0;
        population2[i].ECa = 120.0;
        population2[i].EK  = -84.0;
        population2[i].EL  = -60.0;
        population2[i].Cm  = 20.0;

        /* Synapses */
        population2[i].syn_x2x2.gmax = 0.5;
        population2[i].syn_x2x2.Esyn = -80.0;
        population2[i].syn_x2x2.a    = 5.0;
        population2[i].syn_x2x2.b    = 0.18;
        population2[i].syn_x2x2.Vt   = 2.0;
        population2[i].syn_x2x2.m    = 0.0;

        population2[i].syn_x1x2.gmax = 0.8;
        population2[i].syn_x1x2.Esyn = 0.0; 
        population2[i].syn_x1x2.a    = 1.1;
        population2[i].syn_x1x2.b    = 0.19;
        population2[i].syn_x1x2.Vt   = 2.0;
        population2[i].syn_x1x2.m    = 0.0;

        population2[i].syn_x2x2_slow.gmax = 0.0;   /* example: no slow x2->x2 if you wish */
        population2[i].syn_x2x2_slow.Esyn = -95.0;
        population2[i].syn_x2x2_slow.a    = 0.09;
        population2[i].syn_x2x2_slow.b    = 0.0012;
        population2[i].syn_x2x2_slow.Vt   = 2.0;
        population2[i].syn_x2x2_slow.r    = 0.0;
        population2[i].syn_x2x2_slow.s    = 0.0;
        population2[i].syn_x2x2_slow.Kd   = 100.0;
        population2[i].syn_x2x2_slow.K3   = 0.18;
        population2[i].syn_x2x2_slow.K4   = 0.034;
	//        population2[i].syn_x2x2_slow.n    = 4.0;   //  comment.1 n hardwired to 4

        /* Connection arrays */
        population2[i].pop1in = NULL;
        population2[i].pop1in_count = 0;
        population2[i].pop2in = NULL;
        population2[i].pop2in_count = 0;
        population2[i].pop2in_gj = NULL;
        population2[i].pop2in_gj_count = 0;
    }

    /* Allocate pop1n */
    pop1n *population1 = (pop1n *)malloc(N1 * sizeof(pop1n));
    for(int i=0; i<N1; i++) {
        population1[i].a  = 1.0;
        population1[i].b  = 3.0;
        population1[i].c  = 1.0;
        population1[i].d  = 5.0;
        population1[i].m  = 0.8;
        population1[i].s  = 8.0;
        population1[i].x0 = -2.0;
        population1[i].z0 = 0.0;
        population1[i].r  = 0.0001;
        population1[i].I1 = 3.1;

        population1[i].x1 = uniform_rand(-1.0, 1.5);
        population1[i].y1 = uniform_rand(-5.0, 0.0);
        population1[i].z  = uniform_rand(3.0, 3.0);

        population1[i].CpES = 0.8;  
        population1[i].CpCS = 1.0;  

        population1[i].noise  = 2.0;
        population1[i].noise3 = 0.1;

        /* Fast syn x1->x1 */
        population1[i].syn_x1x1.gmax = 0.0;  /* set to 0 if you don't want x1->x1 */
        population1[i].syn_x1x1.Esyn = 0.0;
        population1[i].syn_x1x1.a    = 1.1;
        population1[i].syn_x1x1.b    = 0.19;
        population1[i].syn_x1x1.Vt   = 2.0;
        population1[i].syn_x1x1.m    = 0.0;

        /* Fast syn x2->x1 */
        population1[i].syn_x2x1.gmax = 0.5;
        population1[i].syn_x2x1.Esyn = uniform_rand(-80.0, -50.0);
        population1[i].syn_x2x1.a    = 5.0;
        population1[i].syn_x2x1.b    = 0.18;
        population1[i].syn_x2x1.Vt   = 2.0;
        population1[i].syn_x2x1.m    = 0.0;

        /* Slow syn x2->x1 */
        population1[i].syn_x2x1_slow.gmax = 1.0;
        population1[i].syn_x2x1_slow.Esyn = -95.0;
        population1[i].syn_x2x1_slow.a    = 0.09;
        population1[i].syn_x2x1_slow.b    = 0.0012;
        population1[i].syn_x2x1_slow.Vt   = 2.0;
        population1[i].syn_x2x1_slow.r    = 0.0;
        population1[i].syn_x2x1_slow.s    = 0.0;
        population1[i].syn_x2x1_slow.Kd   = 100.0;
        population1[i].syn_x2x1_slow.K3   = 0.18;
        population1[i].syn_x2x1_slow.K4   = 0.034;
	//        population2[i].syn_x2x1_slow.n    = 4.0;   //  comment.1 n hardwired to 4

        /* Connection arrays */
        population1[i].pop1in = NULL;
        population1[i].pop1in_count = 0;
        population1[i].pop1in_gj = NULL;
        population1[i].pop1in_gj_count = 0;
        population1[i].pop2in = NULL;
        population1[i].pop2in_count = 0;
    }

    /* For simplicity, connect each pop1n to all other pop1n by gap-junction 
       and to the entire pop2n array by chemical syn. 
       You can tailor these connections as needed. */

    /* Create arrays-of-pointers for each population for connectivity. */
    pop1n **p1_ptrs = (pop1n **)malloc(N1 * sizeof(pop1n*));
    pop2n **p2_ptrs = (pop2n **)malloc(N2 * sizeof(pop2n*));
    for(int i=0; i<N1; i++) { p1_ptrs[i] = &population1[i]; }
    for(int i=0; i<N2; i++) { p2_ptrs[i] = &population2[i]; }

    /* Connect each pop1n to: 
         - all pop1n for gap (pop1in_gj)
         - all pop2n for syn_x2x1 
    */
    for(int i=0; i<N1; i++){
        connect_pop1n_gap(&population1[i], p1_ptrs, N1);       /* gap among pop1n */
        connect_pop1n_syn_pop2n(&population1[i], p2_ptrs, N2); /* chemical from pop2n -> pop1n */
    }

    /* Connect each pop2n to: 
         - all pop2n for gap
         - all pop1n for syn_x1x2
    */
    for(int i=0; i<N2; i++){
        connect_pop2n_gap(&population2[i], p2_ptrs, N2);
        connect_pop2n_syn_pop1n(&population2[i], p1_ptrs, N1);
        connect_pop2n_syn_pop2n(&population2[i], p2_ptrs, N2); 
    }

    /* For demonstration, we'll track the average x1 and x2 at each time step. */
    FILE *fp = fopen("epileptor_output.txt","w");
    if(!fp){
        printf("Cannot open output file.\n");
        return 1;
    }
    fprintf(fp,"# t(ms)\tmean_x1\tmean_x2\n");

    /* Main simulation loop */
    for(int step=0; step<steps; step++){
        double t = step*dt; 

        /* We'll compute the average of x1 and x2 each step, 
           but update neurons in some order. */
        double sum_x1 = 0.0, sum_x2 = 0.0;

        /* For convenience, compute the current average of x1 and x2 
           so that each neuron can see the "bar" values. */
        for(int i=0; i<N1; i++){ sum_x1 += population1[i].x1; }
        double x1bar = (N1>0)? (sum_x1/N1) : 0.0;

        sum_x2 = 0.0;
        for(int i=0; i<N2; i++){ sum_x2 += population2[i].x2; }
        double x2bar = (N2>0)? (sum_x2/N2) : 0.0;

        /* Similarly, for demonstration, let's take an average z among pop1n */
        double sum_z = 0.0;
        for(int i=0; i<N1; i++){ sum_z += population1[i].z; }
        double zbar = (N1>0)? (sum_z/N1) : 0.0;

        /* Update each pop1n with Euler step */
        for(int i=0; i<N1; i++){
            pop1n_euler(&population1[i], dt, step, x1bar, x2bar, zbar);
        }
        /* Update each pop2n with Euler step */
        for(int i=0; i<N2; i++){
            pop2n_euler(&population2[i], dt, step, x1bar, x2bar, zbar);
        }

        /* After updating, compute final new means of x1, x2 just to store/print */
        double new_sum_x1=0.0, new_sum_x2=0.0;
        for(int i=0; i<N1; i++){ new_sum_x1 += population1[i].x1; }
        for(int i=0; i<N2; i++){ new_sum_x2 += population2[i].x2; }
        double mean_x1 = (N1>0)? (new_sum_x1/N1):0.0;
        double mean_x2 = (N2>0)? (new_sum_x2/N2):0.0;

        fprintf(fp, "%g\t%g\t%g\n", t, mean_x1, mean_x2);
    }

    // Misura il tempo di fine
    clock_t end_time = clock();

    // Calcola il tempo di esecuzione in secondi
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione: %f secondi\n", execution_time);

    fclose(fp);
    printf("Simulation finished. Results in epileptor_output.txt\n");

    /* Clean up */
    free(p1_ptrs);
    free(p2_ptrs);
    free(population1);
    free(population2);
    return 0;
}

