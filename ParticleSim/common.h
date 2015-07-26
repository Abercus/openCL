#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct
{
	double x;
	double y;
	double vx;
	double vy;
	double ax;
	double ay;
} particle_t;

//
//  timing routines
//
double read_timer();

//
//  simulation routines
//

void set_size(int n);
void init_particles(int n, particle_t *p);
//void apply_force( particle_t &particle, particle_t &neighbor );
//void move( particle_t &p );

//
//  I/O routines
//
FILE *open_save(char *filename, int n);
void save(FILE *f, int n, particle_t *p);

//
//  argument processing routines
//
int find_option(int argc, char **argv, const char *option);
int read_int(int argc, char **argv, const char *option, int default_value);
char *read_string(int argc, char **argv, const char *option, char *default_value);

#endif
