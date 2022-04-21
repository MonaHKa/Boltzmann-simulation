// Boltzman simulation of a Heisenberg Antiferromagnet with interacting magnons
// by Mona Kalthoff (2022)

// RUN LOCALLY
// mpic++ -O3 -std=c++17 01_Boltzmann_Timeevolution.cpp -o output_Boltz.out
// mpirun -n 2 output_Boltz.out 

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>
#include "boost/lexical_cast.hpp"
#include "boost/format.hpp"

#ifndef NO_MPI
    #include <mpi.h>
#endif
using namespace std;

// Physical Parameters
// MyGrid is the linear system size 
const int MyGrid = 120;

// Run - ComputeQuadruples = true - to save the quadruples that satidfy energy and momentum conservation  
// Run - ConsilidateQuadruples = true - to convert the quadruples into energy quadruples   
// After the Energy Quadruples are saved, both bool variables can be set false to compute the time evolution
const bool ComputeQuadruples = false;
const bool ConsilidateQuadruples = false;

const double Temp=0.6;
const int offset_energy=1;
const double offset_amplitude=0.001; 

// Interaction_switch and Lambda_Scaling determine in how far the drive/the interactions are condidered
// to consider both set both variables = 1 
const double Interaction_switch = 1.0;
const double Lambda_Scaling = 1.0;

/* *****************
 scattering in + scattering out 
 g_in (1+n) - g_out (n+(n/nthermal(Teff))^2)
***************** */
const double GIN = 0.00030;
const double GOUT = 0.0020;
const double Tnoninteracting = 0.6; 


const int dim = 2.0;
const int Z_coord = 4;
const double Delta = 1.0; 
const double Spin = 0.5;
const double hbar = 1.0;
const double Jint = 1.0;
const double OmegaMax=(Jint*Spin*double(Z_coord)*Delta*1.1579474)/hbar;
// Jint depends on the material. Choosing Jint = hbar = 1 defines the timescale, which is 1t=hbar/Jint

const double PI = M_PI;
const double sqrt2PI = sqrt(2.0*PI);
const double MBZ_Volume = pow(sqrt(2.0)*PI,2.0);

const double step = PI / double(MyGrid);
const double dk = sqrt(2.0)*step; // sqrt(2.)*M_PI is length of magnetic BZ (MBZ)
double tolerance = 0.1*dk;

const int NUMEMAX = MyGrid;
const double energystep=OmegaMax/double(NUMEMAX); 
const double ecutoff = 4.*energystep;

const double TMAX = 11000.0;
const double MYDT = 0.00001;
const int NUMT = int(TMAX/MYDT);
//const int PRINTSTEP = int(NUMT/1000);
const int PRINTSTEP = int(0.5/MYDT);

int numprocs=1, myrank=0, namelen;
//This is initialized by the MPI commands and should never be changed.

//-------------------------------------------GENERAL HELP FUNCTIONS
//-----------------------------------------------------------------

template <typename T>
void ReadIn_2d(vector<vector<T>> &MAT, const string &filename) {
  ifstream in(filename,std::ifstream::binary);
  string record;
  bool it_works = true;
  if (in.fail()) {
    cout << "file" << filename << "could not be found!" << endl;
    it_works = false;
  }
  else{
    while (getline(in, record)) {
      istringstream is(record);
      vector<T> row((istream_iterator<T>(is)), istream_iterator<T>());
      MAT.push_back(row);
  }
  }
}
template <typename T>
void WriteIn_2d(const vector<vector<T>> &MAT, const string &filename) {
  auto dummy = filename;
  ofstream myfile(dummy, std::ofstream::binary);
  myfile << setprecision(12);
  if (myfile.is_open()) {
    for (int cc = 0; cc < MAT.size(); cc++) {
      for (int kk = 0; kk < MAT[cc].size(); kk++)
        myfile << MAT[cc][kk] << " ";
      myfile << endl;
    }
    myfile.close();
  } else
    cout << "Unable to open file" << endl;
}
void ReadIn_3d(vector<vector<vector<int>>> &MAT, const string &filename) {
  ifstream in(filename,std::ifstream::binary);
  string record;
  bool it_works = true;
  if (in.fail()) {
    cout << "file" << filename << "could not be found!" << endl;
    it_works = false;
  }
  else{
    while (getline(in, record)) {
      istringstream is(record);
      vector<int> row((istream_iterator<int>(is)), istream_iterator<int>());
      vector<vector<int>> column_vecs;
      int length = row.size();
      if(length % 4 != 0){cout << "Error when reading in " << filename << "!!!" << endl;}
      for(int rr=0; rr<(length/4); rr++){
        column_vecs.push_back( {row[rr*4], row[rr*4+1],row[rr*4+2],row[rr*4+3]} );
      }
      MAT.push_back(column_vecs);
  }
  }
}
void WriteIn_3d(const vector<vector<vector<int>>> &MAT, const string &filename) {
  auto dummy = filename;
  ofstream myfile(dummy, std::ofstream::binary);
  if (myfile.is_open()) {
    for (int rr = 0; rr < MAT.size(); rr++) {
      for (int cc = 0; cc < MAT[rr].size(); cc++){
        for (int thrd = 0; thrd < MAT[rr][cc].size(); thrd++){
          myfile << MAT[rr][cc][thrd] << " ";
        }
      }
      myfile << endl;
    }
    myfile.close();
  } 
  else{ cout << "Unable to open file" << endl;}
}
// Needed in order to print vectors
template <typename T> ostream &operator<<(ostream &out, const vector<T> &v) {
  if (!v.empty()) {
    copy(v.begin(), v.end(), ostream_iterator<T>(out, "  "));
  }
  return out;
}
template <typename T> ostream &operator<<(ostream &out, const vector<vector<T>> &mat) {
  for(auto const& vec : mat){
    if (!vec.empty()) {copy(vec.begin(), vec.end(), ostream_iterator<T>(out, "  "));}
    cout << endl;
  }
  
    
  return out;
}
// + and - need to be overloaded in order to operate for vectors
template <typename T>
vector<T> operator+(const vector<T> &v1, const vector<T> &v2) {
  vector<T> VEC;
  if (v1.size() != v2.size()) {
    cout << "ERROR: Adding two vectors of different size will not work" << endl;
    return {0, 0}; // ASK SOMEONE ABOUT ERRORS
  }
  for (int i = 0; i < v1.size(); i++) {
    VEC.push_back(v1[i] + v2[i]);
  }
  return VEC;
}
template <typename T>
vector<T> operator-(const vector<T> &v1, const vector<T> &v2) {
  vector<T> VEC;
  if (v1.size() != v2.size()) {
    cout << "ERROR: Adding two vectors of different size will not work" << endl;
    return {0, 0}; // ASK SOMEONE ABOUT ERRORS
  }
  for (int i = 0; i < v1.size(); i++) {
    VEC.push_back(v1[i] - v2[i]);
  }
  return VEC;
}
template <typename T>
vector<T> operator*(const vector<T> &v1, const vector<T> &v2) {
  vector<T> VEC;
  if (v1.size() != v2.size()) {
    cout << "ERROR: Multiplying two vectors of different size will not work" << endl;
    return {0, 0}; 
  }
  for (int i = 0; i < v1.size(); i++) {
    VEC.push_back(v1[i] * v2[i]);
  }
  return VEC;
}
template <typename T>
T Abs_Value_Vec(const vector<T> &Vec){
  T abs_val=0.0;
  for (int i = 0; i < Vec.size(); i++) {abs_val=abs_val+Vec[i]*Vec[i];}
  return sqrt(abs_val);
}
template <typename T>
void Normalize_vec(vector<T> &Vec){
  T myabs = Abs_Value(Vec);
  for (int i = 0; i < Vec.size(); i++) {Vec[i]=Vec[i]/myabs;}
}
// Function to give the Minimum and the Position of the Minimum
template <typename T>
void Min_Position(const vector<T> &VEC, int &Index, T &MinimumValue){
  auto min_help = min_element(VEC.begin(),VEC.end());
  Index = distance(VEC.begin(), min_help);
  MinimumValue = VEC[Index];
  if(myrank==0) cout << "Minimum of the Vector at Index --> " << Index << ", Value of the Mimimum --> " << MinimumValue << endl;
}
template <typename T>
void Min_Position(const vector<T> &VEC){
  auto min_help = min_element(VEC.begin(),VEC.end());
  auto Index = distance(VEC.begin(), min_help);
  auto MinimumValue = VEC[Index];
  if(myrank==0) cout << "Minimum of the Vector at Index --> " << Index << ", Value of the Mimimum --> " << MinimumValue << endl;
}
template <typename T>
void Max_Position(const vector<T> &VEC, int &Index, T &MinimumValue){
  auto max_help = max_element(VEC.begin(),VEC.end());
  Index = distance(VEC.begin(), max_help);
  MinimumValue = VEC[Index];
  if(myrank==0) cout << "Maximum of the Vector at Index --> " << Index << ", Value of the Mimimum --> " << MinimumValue << endl;
}
template <typename T>
int Max_Position(const vector<T> &VEC){
  auto max_help = max_element(VEC.begin(),VEC.end());
  auto Index = distance(VEC.begin(), max_help);
  return Index;
}
template <typename T>
double Min_Value(const vector<T> &VEC){
  auto min_help = min_element(VEC.begin(),VEC.end());
  auto Index = distance(VEC.begin(), min_help);
  auto MinimumValue = VEC[Index];
  return MinimumValue;
}
template <typename T>
T Max_Value(const vector<T> &VEC){
  auto max_help = max_element(VEC.begin(),VEC.end());
  auto Index = distance(VEC.begin(), max_help);
  auto MaxValue = VEC[Index];
  return MaxValue;
}
template <typename T>
void MultiplyMatrix(vector<vector<T>> &Mat, const T &Factor){
  //cout << "I am Processor " << myrank << " I am multiplying." << endl;
  for(auto & Vec : Mat){
    //transform(Vec.begin(), Vec.end(), Vec.begin(), [&Factor](auto& c){return c*Factor;});
    for(auto & ent : Vec){
      ent=ent*Factor;}
  }
}
//-----------------------------------------------PHYSICAL FUNCTIONS
//-----------------------------------------------------------------

template<typename T> 
T Gaus(T const & x, double sigma_var){
	return  exp(-x*x/(2*sigma_var*sigma_var)) /(sigma_var*sqrt2PI);
}
template<typename T> 
T vectorlength(const vector<T>& kvec){
	return sqrt(kvec[0]*kvec[0]+kvec[1]*kvec[1]);
}
template<typename T> 
T gamma_f(T const & kx, T const & ky){
	return 0.5*(cos(kx)+cos(ky));
}
template<typename T> 
T gamma_f(const vector<T>& kvec){
	return 0.5*(cos(kvec[0])+cos(kvec[1]));
}
template <typename T>
double Omega(const vector<T>& kvec){
	double gamma_k_dev_delta = gamma_f(kvec)/Delta;
	return sqrt(1.0-(gamma_k_dev_delta*gamma_k_dev_delta))*OmegaMax;
}
template <typename T>
double lambda_f(const vector<T>& kvec){
	double GAMMA= gamma_f(kvec);
	return sqrt(Delta*Delta-GAMMA*GAMMA);
}
template <typename T>
double uk(const vector<T>& kvec){
	double LAMBDA = lambda_f(kvec);
	return sqrt( (Delta+LAMBDA)/ (2.0*LAMBDA) );
}
template <typename T>
double vk(const vector<T>& kvec){
	double LAMBDA = lambda_f(kvec);
		if(gamma_f(kvec)<0){
			return sqrt( (Delta-LAMBDA)/ (2.0*LAMBDA) );
		}
		else {
			return -1.0*sqrt( (Delta-LAMBDA)/ (2.0*LAMBDA) );
		}
}
template <typename T>
double V1234_22(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){
  vector<double> k_text = { PI/3.0, PI/3.0};
  // The Analytic result for k1=k2=k3=k4 is 0 for all Delta
  // The calculation gives results of the magnitude of e-17
	double u1=uk(k1);
	double u2=uk(k2);
	double u3=uk(k3);
	double u4=uk(k4);

	double v1=vk(k1);
	double v2=vk(k2);
	double v3=vk(k3);
	double v4=vk(k4);

	return Delta*gamma_f(k2-k4)*u1*u3*v2*v4 + 0.5*gamma_f(k2)*u1*u3*u4*v2 +0.5*gamma_f(k1)*u1*v2*v3*v4 ;
}
template <typename T>
double V1234_22Tilde(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){
  vector<double> k_text = { PI/3.0, PI/3.0};
  // The Analytic result for k1=k2=k3=k4 is Delta for all Delta
  // The calculation gives the correct result
	double u1=uk(k1);
	double u2=uk(k2);
	double u3=uk(k3);
	double u4=uk(k4);

	double v1=vk(k1);
	double v2=vk(k2);
	double v3=vk(k3);
	double v4=vk(k4);

	return Delta*gamma_f(k2-k4)*(u1*u2*u3*u4 + v1*v2*v3*v4) + Delta*gamma_f(k2-k3)*(u1*u2*v3*v4 + v1*v2*u3*u4) + gamma_f(k2)*(u1*u2*u3*v4 + u4*v1*v2*v3) + gamma_f(k1)*(u1*u2*u4*v3 + u3*v1*v2*v4);
}
// 3:1 Vectors in the HP Formalism
template <typename T>
double V1234_31(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){

	double u1=uk(k1);
	double u2=uk(k2);
	double u3=uk(k3);
	double u4=uk(k4);

	double v1=vk(k1);
	double v2=vk(k2);
	double v3=vk(k3);
	double v4=vk(k4);

  double V = Delta*gamma_f(k2-k4)*(u4*v1*v2*v3+u1*u2*u3*v4);
  V=V+0.25*(gamma_f(k2)*(v1*v2*v3*v4+u1*u2*u3*u4)+gamma_f(k1)*(u1*u2*v3*v4+u3*u4*v1*v2));
  V=V+0.5*gamma_f(k4)*(u2*u4*v1*v3+u1*u3*v2*v4);
	return V;
}
template <typename T>
double V1234_31Tilde(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){

	double u1=uk(k1);
	double u2=uk(k2);
	double u3=uk(k3);
	double u4=uk(k4);

	double v1=vk(k1);
	double v2=vk(k2);
	double v3=vk(k3);
	double v4=vk(k4);

  double V = Delta*gamma_f(k2-k3)*(u1*u3*u4*v2+u2*v1*v3*v4);
  V=V+0.25*(gamma_f(k4)*(u3*u4*v1*v2+u1*u2*v3*v4)+gamma_f(k3)*(u1*u2*u3*u4+v1*v2*v3*v4));
  V=V+0.5*gamma_f(k2)*(u2*u3*v1*v4+u1*u4*v2*v3);
	return V;
}
template <typename T>
double BoseEinst(const vector<T>& k_vec){
  if (Temp<0.0001){return 0;}
  else{
    double EX = exp(- Omega(k_vec)/Temp);
    return EX/(1-EX);
  }
}
template <typename T>
double BoseEinst(const T& Omga){
  if (Temp<0.0001){return 0;}
  else{
    double EX = exp(- Omga/Temp);
    return EX/(1-EX);
  }
}
template <typename T>
double BoseEinst(const T &freq, const T &Teff){
  if (Teff<0.00001){return 0.0;}
  double bose = exp(freq/Teff)-1.0;
  return  (1.0/bose);
}
template <typename T>
double BoseEinst(const T &freq, const T &Teff, const T &mu){
  if (Teff<0.00001){return 0.0;}
  double bose = exp((freq-mu)/Teff)-1.0;
  return  (1.0/bose);
}
template<typename T> 
T Gaus_centered(const vector<T>& k_vec){
  T center = 1.3;
  T sigma_var = 0.3;
  T x = Omega(k_vec)-center;
	return  exp(-x*x/(2*sigma_var*sigma_var)) /(sigma_var*sqrt2PI);
}
template<typename T> 
void compute_noninteracting(const vector<T> &freq, vector<T> &neng_occu){
  // Gives the occupation of the reverse engineering
  // (g-1.0+math.sqrt(abs(1.0+g*(g+2.0+4.0*math.exp(W/T)*(math.exp(W/T)-2.0)))))/(2.0*(math.exp(W/T)-1.0)*(math.exp(W/T)-1.0))
  T gtot=GIN/GOUT;
  T occu;
  for(int i=0; i<freq.size();i++){
    T om = freq[i];
    occu = gtot-1.0+sqrt(abs(1.0+gtot*(gtot+2.0+4.0*exp(om/Tnoninteracting)*(exp(om/Tnoninteracting)-2.0))));
    occu = occu/(2.0*(exp(om/Tnoninteracting)-1.0)*(exp(om/Tnoninteracting)-1.0));
    neng_occu[i] = occu;
  }
}

//-----------------------------------------------FUNCTIONS BOLTZMANN 
//-----------------------------------------------------------------

template <typename T>
void Rotate(const vector<T> &vec, vector<T> &vec_rotated){
  vec_rotated[0] = (1.0/sqrt(2.0))*(vec[0]+vec[1]);
  vec_rotated[1] = (1.0/sqrt(2.0))*(-vec[0]+vec[0]);
}

template <typename T>
void Shift_to_fullzone(vector<T> &k_vec){   
  T kx = k_vec[0];
  T ky = k_vec[1];
  // first rotate into kxtilde, kytilde
  T kxtilde = 1./sqrt(2.)*(kx+ky);
  T kytilde = 1./sqrt(2.)*(-kx+ky);
  double boundary = M_PI/sqrt(2.);
  double shift = sqrt(2.)*M_PI;
  if(kxtilde>boundary) kxtilde -= shift;
  if(kxtilde<-boundary) kxtilde += shift;
  if(kytilde>boundary) kytilde -=shift;
  if(kytilde<-boundary) kytilde +=shift;
  //rotate back
  k_vec[0] = 1./sqrt(2.)*(kxtilde-kytilde);
  k_vec[1] = 1./sqrt(2.)*(kxtilde+kytilde);
}

void Build_MBZ(vector<vector<double>> &MBZ,vector<vector<double>> &MBZ_partzone, 
vector<vector<double>> &MBZ_partzone_position_lengths_Omega, 
vector<int> &Equivalence, vector<double> &kweights, double &kweightsum, vector<double> &Omegas_pz){

 if(myrank==0){ cout << "Building the sorted MBZ for MyGrid --> " << MyGrid << endl;} 
 vector<vector<double>> MBZ_init;
 vector<vector<double>> k_vec;
 const vector<double> k_move_init = {step,step};
 const vector<double> k_move = {step,step};
 vector<double> k_new;
 k_vec.push_back({0.0,-PI +step});
 for(int counter=0; counter<MyGrid-1; counter++){
    k_vec.push_back(k_vec[counter]+k_move);
 }
 for(auto const& kv : k_vec){
   for(int counter=0; counter<MyGrid; counter++){
     k_new = {-step*double(counter), step*double(counter)};
     MBZ_init.push_back(k_new+kv);
   }
 }
 vector<int> sorting_vectorMBZ(MBZ_init.size());
 iota(begin(sorting_vectorMBZ), end(sorting_vectorMBZ), 0);
 sort(sorting_vectorMBZ.begin(),sorting_vectorMBZ.end(), [&MBZ_init](int i, int j){return  vectorlength(MBZ_init[i])<vectorlength(MBZ_init[j]);});
 for(int i=0; i<MBZ_init.size(); i++){MBZ.push_back( MBZ_init[sorting_vectorMBZ[i]]);} 
 // THE MBZ IS SORTED FROM HERE ON

 // Building the partzone:
 double kx, ky, kxp, kyp;
 for(int n_MBZ=0; n_MBZ<MBZ.size(); n_MBZ++){
   kx=MBZ[n_MBZ][0];
   ky=MBZ[n_MBZ][1];
   if(ky>0.0){
     if(kx>-0.5*step && kx<ky){
       MBZ_partzone.push_back(MBZ[n_MBZ]);
       Omegas_pz.push_back(Omega(MBZ[n_MBZ]));
       MBZ_partzone_position_lengths_Omega.push_back({double(n_MBZ),vectorlength(MBZ[n_MBZ]),Omega(MBZ[n_MBZ])});
       if(kx<0.5*step){
         kweights.push_back(1.0);
         kweightsum+=1.0;
        }
        else{
          kweights.push_back(2.0);
          kweightsum+=2.0;
        }
     }
   }
 }
 
 // Finding the equivalence 
 for(int n_MBZ=0; n_MBZ<MBZ.size(); n_MBZ++){
   kx=MBZ[n_MBZ][0];
   ky=MBZ[n_MBZ][1];
   for(int n_MBZ_p=0; n_MBZ_p<MBZ_partzone.size(); n_MBZ_p++){
     kxp=MBZ_partzone[n_MBZ_p][0];
     kyp=MBZ_partzone[n_MBZ_p][1];
     if( (abs(abs(kx)-abs(kxp))<tolerance && abs(abs(ky)-abs(kyp))<tolerance) || (abs(abs(kx)-abs(kyp))<tolerance && abs(abs(ky)-abs(kxp))<tolerance) ){
       Equivalence[n_MBZ] = n_MBZ_p;
     }
   }
 }

 if(myrank==0){ 
   WriteIn_2d(MBZ,"./MyOutput/A0_MBZ_"+ boost::lexical_cast<std::string>(MyGrid) + "MyGrid_Sorted.txt");
   WriteIn_2d(MBZ_partzone,"./MyOutput/A0_MBZ_"+ boost::lexical_cast<std::string>(MyGrid) + "MyGrid_partzone.txt");
   WriteIn_2d(MBZ_partzone_position_lengths_Omega,"./MyOutput/A0_MBZ_"+ boost::lexical_cast<std::string>(MyGrid) + "MyGrid_partzone_position_lengths_Omega_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta.txt");
   cout << "Total Number of k_vectors --> " << MBZ.size() << endl;
   cout << "Partzone Number of k_vectors --> " << MBZ_partzone.size() << endl;
   cout << "kweightsum --> " << kweightsum << endl;}
}

template <typename T>
double Compute_Omega_Max(const vector<vector<T>> &MBZ){
  // Calculate the constant part of Eq. 125
  double Omega_Const=0.0;
  double MBZ_size = double(MBZ.size());
  for(auto const& mbz_vec : MBZ){Omega_Const=Omega_Const+lambda_f(mbz_vec);}
  // The two cancels out because of the two Sublattices, so the N in the denumerator really means MBZ.size*2sublattices
  Omega_Const=1-(Omega_Const)/(MBZ_size);
  Omega_Const=((Jint*Spin*double(Z_coord))/hbar) * (1+(1/(2*Spin)*Omega_Const));
  if(abs(Omega_Const-OmegaMax)>0.01){if(myrank==0)  cout << "ATTENTION: Omega_Const is way to small: abs(Omega_Const-Omegamax) = " << abs(Omega_Const-OmegaMax) << endl;}
  return Omega_Const;
}

int Count_Energies(const vector<vector<double>> &MBZ_partzone, const vector<double> &Omega_pz){
  int NUME=0;
  int partzonesize = MBZ_partzone.size();
  
  int counter = 0;
  int ecounter = 0;

  for(double energy=0.5*energystep; energy<=OmegaMax; energy+=energystep){
    int found =0;
    for(int kpz=0; kpz < partzonesize; kpz++){
      if(abs(Omega_pz[kpz]-energy)<0.5*energystep){found++;}
    }
    if(found>0){
      ecounter++;
    }
  }
  return ecounter;
}

void Conversion_To_EnergyGrid(const vector<vector<double>> &MBZ_partzone, const vector<double> &Omegas_pz, const vector<double> &kweights,
vector<vector<int>> &kforenergies, vector<double> &kweightsforenergies, vector<double> &energies){
  if(myrank==0){ cout << endl <<"Conversion to energy grid" << endl;}
  int NUME=0;
  int partzonesize = MBZ_partzone.size();
  vector<vector<int> > kforenergies_temp(NUMEMAX);
  vector<vector<double> > kweightsforenergies_temp(NUMEMAX);
  
  int counter = 0;
  int ecounter = 0;

  for(double energy=0.5*energystep; energy<=OmegaMax; energy+=energystep){
    for(int kpz=0; kpz < partzonesize; kpz++){
      if(abs(Omegas_pz[kpz]-energy)<0.5*energystep){
        kforenergies_temp[counter].push_back(kpz);
        kweightsforenergies_temp[counter].push_back(kweights[kpz]);
      }
    }
    if(kforenergies_temp[counter].size()>0){
      energies.push_back(energy);
      ecounter++;
    }
    counter++;
  }
  //if(myrank==0){ cout << "You found --> " << ecounter << " Energies that correspond roughly to momentum grid" << endl;}
  if( abs(double(ecounter-kforenergies.size()) )>0.0 ) {if(myrank==0){ cout << "ERROR in Conversion to Energy, wrong ecounter" << endl;}}
  NUME = ecounter;
  ecounter = 0;
  for(int j=0; j<NUMEMAX; j++){
    if(kforenergies_temp[j].size()>0){
      kweightsforenergies[ecounter]=0.0;
      for(int k_count=0; k_count<kforenergies_temp[j].size(); k_count++){
        kforenergies[ecounter].push_back(kforenergies_temp[j][k_count]);
        kweightsforenergies[ecounter]+=kweightsforenergies_temp[j][k_count];
      }
      ecounter++;
    }
  }
  string energies_file = "./MyOutput/A0_MBZ_"+ boost::lexical_cast<std::string>(MyGrid) + "MyGrid_EnergieGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta.txt";
  ofstream Data_EN;
  if(myrank==0){
    Data_EN.open(energies_file);
    Data_EN.precision(10);
    for(int j=0;j<NUME;j++){
      Data_EN << j << "\t" << energies[j];
      //Data_EN << "\t" << kforenergies[j].size(); 
      Data_EN << "\t" << kweightsforenergies[j];
      //Data_EN << "\t" << "those are kn = ";
      //for(int i = 0; i < kforenergies[j].size(); i++){Data_EN << kforenergies[j][i] << " , ";}
      Data_EN << endl;
    }
    cout << "NUMEMAX --> " << NUMEMAX << endl;
    cout << "NUME --> " << NUME << endl;
  }
}

void Assign_Core_to_Momentum(const vector<vector<double>> &MBZ_partzone, const vector<double> &Omegas_pz, const vector<double> &energies, 
vector<int> &core_assigned_to, vector<double> &energiespartzone){
  if(myrank==0){cout << "Assigning a core to each momentum" << endl;}
  for(int pzc=0; pzc< MBZ_partzone.size(); pzc++){
    bool found = false;
    for(int ecount = 0; ecount < energies.size(); ecount++){
      if(abs(energies[ecount]-Omega(MBZ_partzone[pzc]))<0.5*energystep){
        if(found==true){ 
          if(myrank==0){cout << "Error, there seem to be two Energies that correspond to k = " << MBZ_partzone[pzc] << "Here it is E = " << energies[ecount] << endl;}
        }
        found = true;
        core_assigned_to[pzc] = numprocs - 1 - (ecount % numprocs);
        energiespartzone[pzc] = energies[ecount];
      }
    }
  }
  string data_file = "./MyOutput/A0_MBZ_"+ boost::lexical_cast<std::string>(MyGrid) + "MyGrid_Procs.txt";
  ofstream Data;
  if(myrank==0){
    Data.open(data_file);
    Data.precision(10);
    for(int j=0;j<MBZ_partzone.size();j++){
      Data << "knum = " << j << " , Omega = " << Omegas_pz[j] << " , Energy = " << energiespartzone[j]<<" , Core = " << core_assigned_to[j] << endl;
    }
  }
}

template <typename T>
int Find_Position_PZ(const vector<vector<T>> &MBZ_partzone, const vector<T> &Unknown_Vector){
  int Position = -1;
  double searched_length = vectorlength(Unknown_Vector);
  for(int i=0; i<MBZ_partzone.size();i++){
    if(abs(vectorlength(MBZ_partzone[i])-searched_length)<tolerance){
      auto MBZx = MBZ_partzone[i][0];
      auto MBZy = MBZ_partzone[i][1];
      auto Ux=Unknown_Vector[0];
      auto Uy=Unknown_Vector[1];
      if( (abs(abs(MBZx)-abs(Ux))< tolerance && abs(abs(MBZy)-abs(Uy))<tolerance) || (abs(abs(MBZx)-abs(Uy))<tolerance && abs(abs(MBZy)-abs(Ux))<tolerance) ){
        Position=i;
      }
    }
  }
  if(Position<0) cout << "  ERROR:  = Vector --> " << Unknown_Vector << " is not found in the MBZ!" << endl;
  return Position;
}

template <typename T>
double VV_symmetrized(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){
  double vv_sym = 0.125*( V1234_22(k1,k2,k3,k4) + V1234_22(k4,k3,k2,k1) );
  vv_sym = vv_sym +  0.125*( V1234_22(k2,k1,k3,k4) + V1234_22(k4,k3,k1,k2) );
  vv_sym = vv_sym +  0.125*( V1234_22(k2,k1,k4,k3) + V1234_22(k3,k4,k1,k2) );
  vv_sym = vv_sym +  0.125*( V1234_22(k1,k2,k4,k3) + V1234_22(k3,k4,k2,k1) );
  return vv_sym;
}

template <typename T>
double VVtilde_symmetrized(const vector<T>& k1, const vector<T>& k2, const vector<T>& k3, const vector<T>& k4){
  double vv_sym = 0.25*( V1234_22Tilde(k1,k4,k3,k2) + V1234_22Tilde(k3,k2,k1,k4) );
  vv_sym = vv_sym + 0.25*( V1234_22Tilde(k2,k3,k4,k1) + V1234_22Tilde(k4,k1,k2,k3) );
  return vv_sym;
}

void Quadruple_Search(const vector<vector<double>> &MBZ, const vector<vector<double>> &MBZ_partzone, const vector<int> &Equivalence,
vector<vector<vector<int>>> &quadruples, vector<vector<double>> &integrals,
vector<vector<double>> &vertices, const vector<int> &core_assigned_to, const vector<double> &energiespartzone){
  if(myrank==0) cout << endl << "Starting to seach for quadruples" << endl;
  int partzonesize = MBZ_partzone.size();
  int fullzonesize = MBZ.size();
  int totalcount = 0;
  for(int pz1=0; pz1<partzonesize; pz1++){
    if(myrank==core_assigned_to[pz1]){
      if(pz1%10==0) cout << "Doing " << pz1 << " out of " << partzonesize << endl;
      // need to make sure that each core only gets its own energies
      for(int fz2=0; fz2<fullzonesize; fz2++){
        for(int fz3=0; fz3<fullzonesize; fz3++){
          vector<double> k4=MBZ_partzone[pz1]+MBZ[fz2]-MBZ[fz3];
          Shift_to_fullzone(k4);
          int pz2 = Equivalence[fz2];
          int pz3 = Equivalence[fz3];
          int pz4 = Find_Position_PZ(MBZ_partzone,k4);
          // Use the energies from the energy grid, not the true omegas
          double e1 = energiespartzone[pz1];
          double e2 = energiespartzone[pz2];
          double e3 = energiespartzone[pz3];
          double e4 = energiespartzone[pz4];
          double ediffcenter = e1+e2-e3-e4;
          if(abs(ediffcenter) < 0.05*ecutoff){
            double myfactor = 1./pow(double(MyGrid),4)/ecutoff; // division by width of box
            quadruples[pz1].push_back({pz1,pz2,pz3,pz4});
            integrals[pz1].push_back(myfactor);
            totalcount++;
            vector<double> k1 = MBZ_partzone[pz1];
            vector<double> k2 = MBZ[fz2];
            vector<double> k3 = MBZ[fz3];
            double vv_sym = pow(VV_symmetrized(k1,k2,k3,k4),2.0);
            double vvtilde_sym = pow(VVtilde_symmetrized(k1,k2,k3,k4),2.0);
            vertices[pz1].push_back(vv_sym+vvtilde_sym);
          }
        }
      }
    }
    //if(myrank==0){cout << pz1 << " -> I have " << quadruples[pz1].size()<< " quadruples" << endl;}
  }
  #ifndef NO_MPI
    MPI_Allreduce(MPI_IN_PLACE, &totalcount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  #endif
  if(myrank==0){cout << "You found --> " << totalcount << " quadruples." << endl;}
}

void Quadruples_Energy(const vector<vector<int>> &kforenergies, const vector<vector<vector<int>>> &quadruples, const vector<vector<double>> &integrals,
const vector<vector<double>> &vertices, const vector<double> &energies, const vector<double> &Omegas_pz, const vector<double> &kweights, const vector<double> &kweightsforenergies,
vector<vector<vector<int>>> &energyquadruples_consolidated, vector<vector<double>> &energyweights_consolidated){
  if(myrank==0) cout << endl << "Starting to compute quadruples in Energy space" << endl;
  /* **********************************************************************************************
  now we want to turn the quadruple list into an energy list with e1, e2, e3, and e4
  we then want to average for given e1 over the multiple entries
  careful: each rank only knows about part of the quadruples
  strategy: go through k1 in partzone, collect for each core the e1,e2,e3,e4 quadruples
  trap for proper energy conservation here again -- or just assume e4 == e1+e2-e3
  then use MPI communications to condense this into the energy quadruples known to all cores
  ********************************************************************************************** */
  int NUME = energyweights_consolidated.size();
  vector<vector<vector<int>>> energyquadruples(NUME);
  vector<vector<double>> energyweights(NUME); 
  int totalcount = 0;
  for(int ec1=numprocs-1-myrank; ec1<NUME; ec1+=numprocs){
    //cout << "ec1 = " << ec1 << endl;
    for(int kc1=0; kc1<kforenergies[ec1].size(); kc1++){
      // kforenergies contains the partzonenumbers
      int pz1=kforenergies[ec1][kc1];
      for(int j=0; j<quadruples[pz1].size(); j++){
        if(abs(quadruples[pz1][j][0] - pz1)>0.5){cout << "ERROR, something is wrong with the Quadruples" << endl << "pz1 = " << pz1 << " , quadruples[pz1][j][0] = " << quadruples[pz1][j][0] << endl;}
        int pz2=quadruples[pz1][j][1];
        int pz3=quadruples[pz1][j][2];
        int pz4=quadruples[pz1][j][3];
        double myintegral = integrals[pz1][j];
        double vertexsquared = vertices[pz1][j];
        double e1 = energies[ec1];
        double e2 = Omegas_pz[pz2];
        double e3 = Omegas_pz[pz3];
        double e4 = Omegas_pz[pz4];
        int ec2 = 0;
        int ec3 = 0;
        int ec4 = 0;
        for(int i=0; i<NUME; i++){
          if(abs(e2-energies[i])<0.5*energystep){ ec2 = i;}
          if(abs(e3-energies[i])<0.5*energystep){ ec3 = i;}
          if(abs(e4-energies[i])<0.5*energystep){ ec4 = i;}
        }
        if(abs(energies[ec1]+energies[ec2]-energies[ec3]-energies[ec4])<0.01*energystep){
          energyquadruples[ec1].push_back({ec1,ec2,ec3,ec4});
          energyweights[ec1].push_back(myintegral * vertexsquared * kweights[pz1]);
        }
      }
    }
    //cout << "I have ... " << energyweights[ec1].size() << " quadruples for ec1 = " << ec1 << " = " << energies[ec1] << endl;

    /* **********************************************************************************************
    now need to revisit this list to average for given energies[iw1], energies[iw2], energies[iw3]
    this gives a consolidated list of energy quadruples and their weights
    good: no communications needed, this is local for each core
    collect and average the ones for same combination iw1, iw2, iw3
    divide for averaging!
    averaging denominator is known by checking the multiplicity 
    of given iw1 energy = kforenergies[iw].size()
    ********************************************************************************************** */

   for(int ec2=0; ec2<NUME; ec2++){
     for(int ec3=0; ec3<NUME; ec3++){
       int counter23=0;
       for(int wc=0; wc<energyweights[ec1].size(); wc++){
         if(energyquadruples[ec1][wc][1]==ec2 && energyquadruples[ec1][wc][2]==ec3){
           if(counter23==0){
             // this combination of iw2, iw3 has not been found before, now add it
             int ec4=energyquadruples[ec1][wc][3];
             energyquadruples_consolidated[ec1].push_back({ec1,ec2,ec3,ec4});
             energyweights_consolidated[ec1].push_back(energyweights[ec1][wc]/ kweightsforenergies[ec1]);
             // now I have only added the weight for the first instance of iw1, iw2, iw3
           }
            else{energyweights_consolidated[ec1][energyweights_consolidated[ec1].size()-1] += (energyweights[ec1][wc]/ kweightsforenergies[ec1]);} 
            // here I add the other ones
            counter23++;
         }
       }
     }
   }
   totalcount=totalcount+energyweights_consolidated[ec1].size();
   cout << "MyRank " << myrank << " --> Consolidated, I have ... " << energyweights_consolidated[ec1].size() << " quadruples for ec1 = " << ec1 << " = " << energies[ec1] << endl;
 }
 // What does this do?
 #ifndef NO_MPI
  MPI_Allreduce(MPI_IN_PLACE, &totalcount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
 #endif
 if(myrank==0){ cout << "Total Number of energyweights --> " << totalcount << endl; }
}


void Initialize_sublattice_reduced_Thermal(const vector<double> &energies, 
vector<double> &occupations_past, const double &MyTemp){
  //if(myrank==0) cout << endl << "Initializing the sublattices with a Thermal distribution at temperature T = " << Temp << endl;
  //if(myrank==0) cout << endl << "There is no Perturbation" << endl;
  for(int i=0; i<energies.size();i++){
    double freq = energies[i];
    double Teff = MyTemp;
    auto occu = BoseEinst(freq, Teff);
    occupations_past[i]=occu;
  }
}

void Initialize_sublattice_reduced_Thermal(const vector<double> &energies, 
vector<double> &occupations_present, vector<double> &occupations_past, const double &MyTemp){
  if(myrank==0) cout << endl << "Initializing the sublattices with a Thermal distribution at temperature T = " << MyTemp << endl;
  for(int i=0; i<energies.size();i++){
    occupations_present[i]=0.0;
    double freq = energies[i];
    double Teff = MyTemp;
    auto occu = BoseEinst(freq, Teff);
    occupations_past[i]=occu;
  }
}

void Initialize_sublattice_reduced_Mu(const vector<double> &energies, 
vector<double> &occupations_past, const double &MyTemp, const double &MyMu){
  //if(myrank==0) cout << endl << "Initializing the sublattices with a Thermal distribution at temperature T = " << Temp << endl;
  //if(myrank==0) cout << endl << "There is no Perturbation" << endl;
  for(int i=0; i<energies.size();i++){
    double freq = energies[i];
    double Teff = MyTemp;
    auto occu = BoseEinst(freq, Teff, MyMu);
    occupations_past[i]=occu;
  }
}

void Initialize_sublattice_Noninteracting(const vector<double> &energies, 
vector<double> &occupations_present, vector<double> &occupations_past){
    if(myrank==0){ cout << endl << "Initializing with Reverse Engineered Solution" << endl;}
    for(int i=0; i<energies.size();i++){occupations_present[i]=0.0;}
    compute_noninteracting(energies, occupations_past);
}

double Find_Energy_reduced(const vector<double> &MyEnergies, const vector<double> &occu, const vector<double> &kweightsforenergies, const double &kweightsum){
  vector<double> myProducts=MyEnergies*occu;
  myProducts=myProducts*kweightsforenergies;
  double myEnergy=accumulate(myProducts.begin(),myProducts.end(),0.0);
  return (myEnergy/kweightsum);// division to get overall "DOS" normalized
}

double Find_ParticleNumber(const vector<double> &occu, const vector<double> &kweightsforenergies, const double &kweightsum){
  vector<double> myProducts=kweightsforenergies*occu;
  double myPNumber=accumulate(myProducts.begin(),myProducts.end(),0.0);
  return (myPNumber/kweightsum);// division to get overall "DOS" normalized
}

double Find_staggered_magnetization(const vector<double> &MyEnergies, const vector<double> &occu, const vector<double> &kweightsforenergies, const double &kweightsum){
  double Constfac = (Spin+0.5); 
  double mag=0.0;
  for(int ec = 0; ec<MyEnergies.size();ec++){
    mag = mag + ((kweightsforenergies[ec]/MyEnergies[ec])*(occu[ec]+0.5)); 
  }
  mag=mag/kweightsum;
  mag=mag*OmegaMax;
  //cout << "Staggered Magnetization = " << (Constfac - mag) << endl;
  return (Constfac-mag);
}

void Find_Energy_at_Frequency(vector<double> &Energy_at_frequency, const int &NUME, const vector<double> &MyEnergies, const vector<double> &occu, const vector<double> &MyDOS){
  for(int i=0; i<NUME; i++){
    Energy_at_frequency[i]=MyEnergies[i]*occu[i]*MyDOS[i]; 
  }
}

void Timeevolution_reduced(const int &NUME, const vector<double> &energies, const vector<double> &kweightsforenergies, const double &kweightsum,
const vector<vector<vector<int>>> &energyquadruples_consolidated, const vector<vector<double>> &energyweights_consolidated){
  if(myrank==0) cout << endl << "Starting with the time stepping" << endl;
  vector<double> occupations_present(NUME);
  vector<double> occupations_past(NUME);
  vector<double> scattering_integrals_present(NUME);
  vector<double> scattering_integrals_past(NUME);

  double dt = TMAX/NUMT;

  // Normed DOS, Computed on a 2000x2000 Grid (minimizing noise)
  vector<double> Energy_at_frequency(NUME);
  vector<vector<double>> DOS_Data;
  ReadIn_2d(DOS_Data,"./MyQuads/01_kweights_reduced_2000_to_"+ boost::lexical_cast<string>(MyGrid) +".txt");
  vector<double> MyDOS(DOS_Data.size());
  for(int i=0; i<DOS_Data.size(); i++){MyDOS[i]=DOS_Data[i][1];}
  string energy_frequ_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_ENERGY_AT_FREQ_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  ofstream Data_EN_F;
  if(myrank==0){
    Data_EN_F.open(energy_frequ_file);
    Data_EN_F.precision(10);
  }

  string scat_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_ScatteringIntegrals_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp_"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  string occu_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_OCCUPATION_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp_"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  string energy_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_ENERGY_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp_"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  string pnumber_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_PNUMBER_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp_"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  string mag_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_MAGNETIZATION_"+ boost::lexical_cast<string>(boost::format("%.2f") %offset_energy) + "eoff_"+ boost::lexical_cast<string>(boost::format("%.4f") %dt) + "dt_"+ boost::lexical_cast<string>(boost::format("%.4f") %offset_amplitude) + "offamp_"+ boost::lexical_cast<string>(boost::format("%.4f") %Temp) + "T_"+ boost::lexical_cast<string>(boost::format("%.6f") %GIN) + "GIN_"+ boost::lexical_cast<string>(boost::format("%.4f") %GOUT) + "GOUT_"+ boost::lexical_cast<string>(boost::format("%.4f") %Tnoninteracting) + "TN_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(boost::format("%.1f") %Interaction_switch) + "INT.txt";
  // ===============================
  ofstream Data_OCCU;
  ofstream Data_SCAT;
  ofstream Data_EN;
  ofstream Data_PN;
  ofstream Data_MAG;
  if(myrank==0){
	  Data_OCCU.open(occu_file);
    Data_OCCU.precision(10);
    Data_SCAT.open(scat_file);
    Data_SCAT.precision(10);
    Data_EN.open(energy_file);
    Data_EN.precision(10);
    Data_PN.open(pnumber_file);
    Data_PN.precision(10);
    Data_MAG.open(mag_file);
    Data_MAG.precision(10);
  }
  // ===============================

  // Initialize the system
  // Initialize_sublattice_reduced(energies, occupations_present, occupations_past);
  Initialize_sublattice_Noninteracting(energies, occupations_present, occupations_past);
  

  double  En_total = Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum); 
  double  PNumber = Find_ParticleNumber(occupations_past, kweightsforenergies, kweightsum); 
  double  Stag_Mag = Find_staggered_magnetization(energies, occupations_past, kweightsforenergies, kweightsum);
  if(MyGrid==120){Find_Energy_at_Frequency(Energy_at_frequency, NUME, energies, occupations_past, MyDOS);}
  if(myrank==0){
    Data_OCCU << 0.0 << " " << energies << endl << 0.0 << " " << occupations_past << endl;
    //for( int i=0; i<NUME; i++){Data_OCCU << 0.0 << " " << energies[i] << " " << occupations_past[i] << " " << kweightsforenergies[i] << endl;}
    Data_EN << 0.0 << " " << En_total << endl;
    Data_EN_F << 0.0 << " " << energies << endl << 0.0 << " " << Energy_at_frequency << endl;
    Data_PN << 0.0 << " " << PNumber << endl;
    Data_MAG << 0.0 << " " << Stag_Mag << endl;
    cout << "Time Evolution with dt --> " << dt << " and tmax --> " << TMAX << endl;  
    cout << "GIN --> " << GIN << endl;
    cout << "GOUT --> " << GOUT << endl;
    cout << "TNicklas --> " << Tnoninteracting << endl;
    cout << "Interaction switch --> " << Interaction_switch << endl;
  }

  // ***************************** // 
  // First time step
  // ***************************** //
  for(int ec1=numprocs-myrank-1; ec1<NUME; ec1+=numprocs){
    scattering_integrals_present[ec1]= 0.0;
    for(int j=0; j<energyweights_consolidated[ec1].size(); ++j){
      if(abs(ec1-energyquadruples_consolidated[ec1][j][0])>0.5){cout << "ERROR, something wrong with quadruples, ec1 = " << ec1 << "quad[0] = " << energyquadruples_consolidated[ec1][j][0] << endl;}
      // Form: energyquadruples_consolidated[energy_counter][number_quadruple][ec1,ec2,ec3,ec4]
      // Form: energyweights_consolidated[energy_counter][vertices with the appropriete weights]
      int ec2 = energyquadruples_consolidated[ec1][j][1];
      int ec3 = energyquadruples_consolidated[ec1][j][2];
      int ec4 = energyquadruples_consolidated[ec1][j][3];
      double n1 = occupations_past[ec1]; 
      double n2 = occupations_past[ec2]; 
      double n3 = occupations_past[ec3]; 
      double n4 = occupations_past[ec4]; 
      double scattering_term = energyweights_consolidated[ec1][j] * ((1.0+n1)*(1.0+n2)*n3*n4 - n1*n2*(1.0+n3)*(1.0+n4));
      scattering_integrals_present[ec1] += scattering_term;
    }
    scattering_integrals_present[ec1] *= 2.*M_PI*pow(double(Z_coord),2.0);
    occupations_present[ec1] = occupations_past[ec1] + dt*scattering_integrals_present[ec1];

    // Include drive/reservoir
    // gin*(1+n)-gout*(n+(n/neff)^2)
    // neff = 1/e^(Omega/T_Nonint)-1

    double neff = 1.0/(exp(energies[ec1]/Tnoninteracting)-1.0);
    double Gtotal = GIN*(1.0+occupations_past[ec1])-GOUT*(occupations_past[ec1]+pow(occupations_past[ec1]/neff,2.0));
    Gtotal=Gtotal*Interaction_switch;
    occupations_present[ec1] += dt*Gtotal;
  }
  #ifndef NO_MPI
    MPI_Allreduce(MPI_IN_PLACE, &occupations_present[0], NUME, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif

  for(int ec1=0; ec1<NUME; ec1++){
    occupations_past[ec1] = occupations_present[ec1]; 
    scattering_integrals_past[ec1] = scattering_integrals_present[ec1];
    occupations_present[ec1] = 0.0;
    scattering_integrals_present[ec1] = 0.0; 
  }
  En_total=Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum);
  PNumber=Find_ParticleNumber(occupations_past, kweightsforenergies, kweightsum); 
  Stag_Mag = Find_staggered_magnetization(energies, occupations_past, kweightsforenergies, kweightsum);
  if(MyGrid==120){Find_Energy_at_Frequency(Energy_at_frequency, NUME, energies, occupations_past, MyDOS);}
  if(myrank==0){
    Data_OCCU << dt << " " << occupations_past << endl;
    Data_EN_F << dt << " " << Energy_at_frequency << endl;
    //for( int i=0; i<NUME; i++){Data_OCCU << 0.0 << " " << energies[i] << " " << occupations_past[i] << " " << kweightsforenergies[i] << endl;}
    Data_EN << dt << " " << En_total << endl;
    Data_PN << dt << " " << PNumber << endl;
    Data_MAG << dt << " " << Stag_Mag << endl;
  }

  //*******************************************************************//
  // Timesteps: Linear multistep method --> Two-step Adams–Bashforth
  //*******************************************************************//

  for(int tc=2; tc<NUMT; tc++){
    double time = double(tc)*dt;
    for(int ec1=numprocs-myrank-1; ec1<NUME; ec1+=numprocs){
      scattering_integrals_present[ec1]= 0.0;
      for(int j=0; j<energyweights_consolidated[ec1].size(); ++j){
        if(abs(ec1-energyquadruples_consolidated[ec1][j][0])>0.5){cout << "ERROR, something wrong with quadruples, ec1 = " << ec1 << "quad[0] = " << energyquadruples_consolidated[ec1][j][0] << endl;}
        // Form: energyquadruples_consolidated[energy_counter][number_quadruple][ec1,ec2,ec3,ec4]
        // Form: energyweights_consolidated[energy_counter][vertices with the appropriete weights]
        int ec2 = energyquadruples_consolidated[ec1][j][1];
        int ec3 = energyquadruples_consolidated[ec1][j][2];
        int ec4 = energyquadruples_consolidated[ec1][j][3];
        double n1 = occupations_past[ec1]; 
        double n2 = occupations_past[ec2]; 
        double n3 = occupations_past[ec3]; 
        double n4 = occupations_past[ec4]; 
        double scattering_term = energyweights_consolidated[ec1][j] * ((1.0+n1)*(1.0+n2)*n3*n4 - n1*n2*(1.0+n3)*(1.0+n4));
        scattering_integrals_present[ec1] += scattering_term;
      }
      scattering_integrals_present[ec1] *= 2.*M_PI*pow(double(Z_coord),2.0);
      occupations_present[ec1] = occupations_past[ec1] + 1.5*dt*scattering_integrals_present[ec1] + 0.5*dt*scattering_integrals_past[ec1];

      // Include drive/reservoir
      // gin*(1+n)-gout*(n+(n/neff)^2)
      // neff = 1/e^(Omega/T_Nonint)-1

      double neff = 1.0/(exp(energies[ec1]/Tnoninteracting)-1.0);
      double Gtotal = GIN*(1.0+occupations_past[ec1])-GOUT*(occupations_past[ec1]+pow(occupations_past[ec1]/neff,2.0));
      Gtotal=Gtotal*Interaction_switch;
      occupations_present[ec1] += dt*Gtotal; 
    }
    #ifndef NO_MPI
      MPI_Allreduce(MPI_IN_PLACE, &occupations_present[0], NUME, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &scattering_integrals_present[0], NUME, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    for(int ec1=0; ec1<NUME; ec1++){
      occupations_past[ec1] = occupations_present[ec1]; 
      scattering_integrals_past[ec1] = scattering_integrals_present[ec1];
      occupations_present[ec1] = 0.0;
      scattering_integrals_present[ec1] = 0.0; 
    }
    PNumber=Find_ParticleNumber(occupations_past, kweightsforenergies, kweightsum); 
    En_total=Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum);
    Stag_Mag = Find_staggered_magnetization(energies, occupations_past, kweightsforenergies, kweightsum);
    if(MyGrid==120){Find_Energy_at_Frequency(Energy_at_frequency, NUME, energies, occupations_past, MyDOS);}
    if(myrank==0){
      if(tc%PRINTSTEP==0){
        Data_OCCU << time << " " << occupations_past << endl;
        Data_SCAT << time << " " << scattering_integrals_past << endl;
        Data_EN_F << time << " " << Energy_at_frequency << endl;
        Data_EN << time << " " << En_total << endl;
        Data_PN << time << " " << PNumber << endl;
        Data_MAG << time << " " << Stag_Mag << endl;
        }
      //Data_PN << time << " " << PNumber << endl;
    }
  }
  if(myrank==0) {cout << "\n ... done! \n"<< endl;}
}

void FindScatteringMatrix(const int &NUME, const vector<double> &energies, 
const vector<vector<vector<int>>> &energyquadruples_consolidated, const vector<vector<double>> &energyweights_consolidated, const vector<double> &kweightsforenergies, const double &kweightsum){
  if(myrank==0) cout << endl << "Starting with the time stepping" << endl;
  vector<double> occupations_past(NUME);
  vector<double> scattering_integrals_present(NUME);

  double offset=0.001;
  double multiply=1.0;
  double mytemp=0.6;
  string scat_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_ScatteringIntegrals_"+boost::lexical_cast<string>(boost::format("%.2f") %mytemp)+ "MyTemp_"+boost::lexical_cast<string>(boost::format("%.2f") %multiply)+"multiply"+boost::lexical_cast<string>(boost::format("%.4f") %offset) + "offset_"+boost::lexical_cast<string>(boost::format("%.0f") %numprocs) +"procs.txt";
  string occp_file = "./MyOutput/A3_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_occupation_"+boost::lexical_cast<string>(boost::format("%.2f") %mytemp)+ "MyTemp_"+boost::lexical_cast<string>(boost::format("%.2f") %multiply)+"multiply"+boost::lexical_cast<string>(boost::format("%.4f") %offset) + "offset_"+boost::lexical_cast<string>(boost::format("%.0f") %numprocs) +"procs.txt";
  
  // ===============================
  
  ofstream Data_SCAT;
  ofstream Data_OC;
  if(myrank==0){
    Data_SCAT.open(scat_file);
    Data_SCAT.precision(10);
    Data_OC.open(occp_file);
    Data_OC.precision(10);
  }
  Initialize_sublattice_reduced_Thermal(energies, occupations_past, mytemp);
  double En_total=Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum);
  if(myrank==0){cout << endl << "Energy Thermal = " << En_total << endl;}
  // ***************************** // 
  // First time step
  // ***************************** //
  for(int energy_off=0;energy_off<NUME;energy_off++){
    // Initialize the system
    // Initialize_sublattice_reduced(energies, occupations_present, occupations_past);
    Initialize_sublattice_reduced_Thermal(energies, occupations_past, mytemp);
    occupations_past[energy_off]+=offset;
    En_total=Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum);
    if(myrank==0){cout << "Energy shift w: " << energy_off << " --> " << En_total << ", n(w) = "<< occupations_past[energy_off] << endl;}
    for(int ec1=numprocs-myrank-1; ec1<NUME; ec1+=numprocs){
      scattering_integrals_present[ec1]= 0.0;
      for(int j=0; j<energyweights_consolidated[ec1].size(); ++j){
        if(abs(ec1-energyquadruples_consolidated[ec1][j][0])>0.5){cout << "ERROR, something wrong with quadruples, ec1 = " << ec1 << "quad[0] = " << energyquadruples_consolidated[ec1][j][0] << endl;}
        // Form: energyquadruples_consolidated[energy_counter][number_quadruple][ec1,ec2,ec3,ec4]
        // Form: energyweights_consolidated[energy_counter][vertices with the appropriete weights]
        int ec2 = energyquadruples_consolidated[ec1][j][1];
        int ec3 = energyquadruples_consolidated[ec1][j][2];
        int ec4 = energyquadruples_consolidated[ec1][j][3];
        double n1 = occupations_past[ec1]; 
        double n2 = occupations_past[ec2]; 
        double n3 = occupations_past[ec3]; 
        double n4 = occupations_past[ec4]; 
        double scattering_term = energyweights_consolidated[ec1][j] * ((1.0+n1)*(1.0+n2)*n3*n4 - n1*n2*(1.0+n3)*(1.0+n4));
        scattering_integrals_present[ec1] += scattering_term;
      }
    scattering_integrals_present[ec1] *= 2.*M_PI*pow(double(Z_coord),2.0);
    }
    #ifndef NO_MPI
      MPI_Allreduce(MPI_IN_PLACE, &scattering_integrals_present[0], NUME, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    if(myrank==0){
        Data_SCAT  << scattering_integrals_present << endl;
        Data_OC  << occupations_past << endl;
    }
    for(int ec1=0; ec1<NUME; ec1++){
      occupations_past[ec1] = 0.0; 
      scattering_integrals_present[ec1] = 0.0;
    }
  }
}

void Find_critical_Pnumber(const int &NUME, const vector<double> &energies, const vector<double> &kweightsforenergies, const double &kweightsum){
  vector<double> occupations_past(NUME);
  double  En_total;
  double  PNumber;
  double Stag_Mag;
  string Myfile_p = "./MyOutput/A3_"+ boost::lexical_cast<std::string>(MyGrid) + "Find_critical_Pnumber_Energy.txt";
  ofstream Data_p;
  if(myrank==0){
    Data_p.open(Myfile_p);
    Data_p << endl; 
    for(double MyT=0.0; MyT<1.0; MyT=MyT+0.004){
        Initialize_sublattice_reduced_Thermal(energies, occupations_past, MyT);
        PNumber=Find_ParticleNumber(occupations_past, kweightsforenergies, kweightsum); 
        En_total=Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum); 
        Data_p << MyT << " " << PNumber << " " <<  En_total << endl;
    }
    Data_p.close();
  cout << "Find_critical_Pnumber is finished" << endl << endl;
  }
}

// #############################################   MAIN

int main(int argc, char * argv[]) {

  //************** MPI INIT ***************************
  
  #ifndef NO_MPI
  	  char processor_name[MPI_MAX_PROCESSOR_NAME];
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	MPI_Get_processor_name(processor_name, &namelen);
	  cout << "Process " << myrank << " on " << processor_name << " out of " << numprocs << " says hello." << endl;
	  MPI_Barrier(MPI_COMM_WORLD);
	  MPI_Datatype ttype  = MPI_DOUBLE;
  #endif
	  if(myrank==0) cout << "\n\tProgram running on " << numprocs << " processors." << endl;
    if(myrank==0) cout << "------------------------------------------" << endl;
  #ifndef NO_MPI
	  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  if(myrank==0) cout << endl << "Main-Function Start" << endl << endl;
  const clock_t begin_time = clock();

  if(myrank==0){ 
    if(ComputeQuadruples==true) cout << "You will compute the quadruples, not read them in !!" << endl;
    else cout << "You will read in the quadruples" << endl;
    cout << "Temperature --> " << Temp << endl;
    cout << "Omegamax --> " << OmegaMax << endl << endl;    
  } 
  
  /************************************************
  ***** Building the (reduced) MBZ
  *************************************************/

  int fullzonesize = int(MyGrid*MyGrid);
  vector<vector<double>> MBZ;
  vector<vector<double>> MBZ_partzone;
  vector<vector<double>> MBZ_partzone_position_lengths_Omega;
  vector<double> Omegas_pz;
  vector<double> kweights;
  vector<int> Equivalence(fullzonesize);
  double kweightsum = 0.;
  Build_MBZ(MBZ, MBZ_partzone, MBZ_partzone_position_lengths_Omega, Equivalence, kweights, kweightsum, Omegas_pz);
  int partzonesize = MBZ_partzone.size();
  if(myrank==0){
    cout << endl << "Time : " << float(clock() - begin_time) / CLOCKS_PER_SEC; 
    cout << " seconds = " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0; 
    cout << " minutes " << endl;}
  
  #ifndef NO_MPI
	  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  if(myrank==0){ cout << endl << "Finding the Corredponding Energy Grid" << endl << endl;}
  /************************************************
  ***** Finding the Corredponding Energy Grid
  *************************************************/
  
  int NUME = Count_Energies(MBZ_partzone, Omegas_pz); 
  vector<vector<int> > kforenergies(NUME);
  // kforenergies[energy][partzonenumber]
  vector<double>  kweightsforenergies(NUME);
  vector<double> energies;
  Conversion_To_EnergyGrid(MBZ_partzone, Omegas_pz, kweights, kforenergies, kweightsforenergies, energies);
  
  vector<int> core_assigned_to(partzonesize);
  vector<double> energiespartzone(partzonesize); 
  Assign_Core_to_Momentum(MBZ_partzone, Omegas_pz, energies, core_assigned_to, energiespartzone);
  if(myrank==0){cout << endl << "Time : " << float(clock() - begin_time) / CLOCKS_PER_SEC << " seconds = " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0 << " minutes " << endl;}
  
  #ifndef NO_MPI
	  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  /************************************************
  ***** Searching for Momentum-Energy-Quadruples
  *************************************************/
  
  // This is just for bookkeeping and will not be used later:  
  if(myrank==1){
    vector<double> occupations_present(NUME);
    vector<double> occupations_past(NUME);
    Initialize_sublattice_Noninteracting(energies, occupations_present, occupations_past);
    cout << "System Energy =" << Find_Energy_reduced(energies, occupations_past, kweightsforenergies, kweightsum) << endl;
    cout << "Staggered mag. =" << Find_staggered_magnetization(energies, occupations_past, kweightsforenergies, kweightsum) << endl;
    cout << "Particle Number =" << Find_ParticleNumber(occupations_past, kweightsforenergies, kweightsum) << endl;
  }
  
  if(ComputeQuadruples==true){
    if(myrank==0){cout << endl << "Searching for Quadruples" << endl << endl;}
    vector<vector<vector<int>>> quadruples(partzonesize);
    // Form: quadruples[partzonevec][number of quad][k1,k2,k3,k4]
    vector<vector<double>> integrals(partzonesize); 
    vector<vector<double>> vertices(partzonesize);
    Quadruple_Search(MBZ, MBZ_partzone, Equivalence, quadruples, integrals, vertices, core_assigned_to, energiespartzone);
    if(myrank==0){cout << endl << "Time : " << float(clock() - begin_time) / CLOCKS_PER_SEC << " seconds = " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0 << " minutes " << endl;}
    WriteIn_3d(quadruples, "./MyQuads/Quadruples_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    WriteIn_2d(integrals, "./MyQuads/Integrals_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    WriteIn_2d(vertices, "./MyQuads/Vertices_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    // cout << "MyRank --> " << myrank << " quadruples[1,2] = " << quadruples[1,2] << endl;
    // cout << "MyRank --> " << myrank << " -> I have " << quadruples[1].size()<< " quadruples[1]" << endl;
    // cout << "MyRank --> " << myrank << " -> I have " << quadruples[0].size()<< " quadruples[0]" << endl;
    #ifndef NO_MPI
	  MPI_Barrier(MPI_COMM_WORLD);
    #endif
  }
  else if(ConsilidateQuadruples==true){
    if(myrank==0){cout << endl << "Consolidating Quadruples" << endl << endl;}
    vector<vector<vector<int>>> quadruples;
    // Form: quadruples[partzonevec][number of quad][k1,k2,k3,k4]
    vector<vector<double>> integrals; 
    vector<vector<double>> vertices;
    vector<vector<vector<int>>> energyquadruples_consolidated(NUME);
    vector<vector<double>> energyweights_consolidated(NUME);
    // Form: energyquadruples_consolidated[energy_counter][number_quadruple][ec1,ec2,ec3,ec4]
    // Form: energyweights_consolidated[energy_counter][vertices with the appropriete weights]
    ReadIn_3d(quadruples, "./MyQuads/Quadruples_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    ReadIn_2d(integrals, "./MyQuads/Integrals_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    ReadIn_2d(vertices, "./MyQuads/Vertices_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    // cout << "MyRank --> " << myrank << " quadruples[1,2] = " << quadruples[1,2] << endl;
    // cout << "MyRank --> " << myrank << " -> I have " << quadruples[1].size()<< " quadruples[1]" << endl;
    // cout << "MyRank --> " << myrank << " -> I have " << quadruples[0].size()<< " quadruples[0]" << endl;
    Quadruples_Energy(kforenergies, quadruples, integrals, vertices, energies, Omegas_pz, kweights, kweightsforenergies, energyquadruples_consolidated, energyweights_consolidated);
    if(myrank==0){cout << endl << "Time : " << float(clock() - begin_time) / CLOCKS_PER_SEC << " seconds = " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0 << " minutes " << endl;}
    WriteIn_2d(energyweights_consolidated, "./MyQuads_E/Eweights_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    WriteIn_3d(energyquadruples_consolidated, "./MyQuads_E/EQuads_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
  }
  else{
    #ifndef NO_MPI
	    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    if(myrank==0){cout << "Careful, you are reading the quadruples in from a previous calculation!!" << endl << endl;}
    //cout << "Process " << myrank << " on " << processor_name << " starts reading in Quads " << endl;
    vector<vector<double>> energyweights_consolidated;
    vector<vector<vector<int>>> energyquadruples_consolidated;
    ReadIn_2d(energyweights_consolidated, "./MyQuads_E/Eweights_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    ReadIn_3d(energyquadruples_consolidated,"./MyQuads_E/EQuads_"+ boost::lexical_cast<string>(MyGrid) + "MyGrid_"+ boost::lexical_cast<string>(boost::format("%.4f") %Delta) +"Delta_"+ boost::lexical_cast<string>(myrank) + "_MyRank.txt");
    MultiplyMatrix(energyweights_consolidated,Lambda_Scaling*Lambda_Scaling);
    if(myrank==0){cout << "Multiplication with Lambda_Scaling*Lambda_Scaling = " << Lambda_Scaling*Lambda_Scaling << endl;}
    Timeevolution_reduced(NUME,energies,kweightsforenergies, kweightsum, energyquadruples_consolidated, energyweights_consolidated);
    //FindScatteringMatrix(NUME,energies,energyquadruples_consolidated,energyweights_consolidated, kweightsforenergies, kweightsum);
  }
  
  if(myrank==0) {
    cout << endl << "The Calculations lasted: " << float(clock() - begin_time) / CLOCKS_PER_SEC << " seconds" << endl;
    cout << "This is equal to: " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0 << " minutes" << endl;
    cout << "This is equal to: " << (float(clock() - begin_time) / CLOCKS_PER_SEC)/60.0/60.0 << " hours" << endl;
    cout << endl << "Main-Function End" << endl; 
  }
  
 
  #ifndef NO_MPI
	  MPI_Finalize();
  #endif
  
  return 0;
}
