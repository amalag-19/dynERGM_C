// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
using namespace arma;
using namespace std;

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
vec rowsum_Mat(mat M) {
    int nr=M.n_rows;
    vec out(nr);
    for(int i=0;i<nr;i++){
        out(i)=sum(M.row(i));
    }
    return out;
}

// [[Rcpp::export]]
vec colsum_Mat(mat M) {
    int nc=M.n_cols;
    vec out(nc);
    for(int i=0;i<nc;i++){
        out(i)=sum(M.col(i));
    }
    return out;
}

// [[Rcpp::export]]
float epan(float input){
    float output;
    if(abs(input)<=1){
        output=0.75*(1-pow(input,2));
    }
    else{
        output=0;
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gamma, Tau update function, gradient and hessian functions and ELBO convergence function

// [[Rcpp::export]]
mat gamma_init_update_HMM_undir(mat gamma_init, vec log_alpha, vec theta, mat network_init, int N, int K){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    mat gamma_init_next(N,K);
    for(int i = 0; i < N; i++){
        vec log_gamma_init_i=log_alpha;
        for(int k = 0; k < K; k++){
            for(int j = 0; j < N; j++){
                for(int l = 0; l < K; l++){
                    log_gamma_init_i(k)+=gamma_init(j,l)*((network_init(i,j)*(theta(k)+theta(l)))-log(1+exp_val_mat(k,l)));
                }
            }
        }
        float log_gamma_init_i_max=log_gamma_init_i.max();
        for(int k = 0; k < K; k++){
            gamma_init_next(i,k)=exp(log_gamma_init_i(k)-log_gamma_init_i_max);
        }
    }
    return gamma_init_next;
}

// [[Rcpp::export]]
cube gamma_update_HMM_undir(mat gamma_init, cube Tau, int N, int K, int T_data){
    cube gamma_next(N,K,T_data,fill::zeros);
    gamma_next.slice(0)=gamma_init;
    for(int t = 1; t < T_data; t++){
        for(int i = 0; i < N; i++){
            for(int k = 0; k < K; k++){
                for(int k_dash = 0; k_dash < K; k_dash++){
                    gamma_next(i,k,t)+=(gamma_next(i,k_dash,t-1)*Tau(k_dash,k,N*(t-1)+i));
                }
            }
        }
    }
    return gamma_next;
}

// [[Rcpp::export]]
field<mat> Tau_update_HMM_undir(cube gamma, cube log_pi, vec theta, cube network, int N, int K, int T_data){
    field<mat> Tau_next(N*(T_data-1),1);
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    
    int increment_index=0;
    for(int t = 0; t < (T_data-1); t++){
        cube Tau_next_t(K,K,N);
        for(int i = 0; i < N; i++){
            mat log_Tau_i=log_pi.slice(t);
            for(int k = 0; k < K; k++){
                for(int k_dash = 0; k_dash < K; k_dash++){
                    for(int j = 0; j < N; j++){
                        if(j!=i){
                            for(int l = 0; l < K; l++){
                                log_Tau_i(k,k_dash)+=gamma(j,l,t)*((network(i,j,t+1)*(theta(k_dash)+theta(l)))-log(1+exp_val_mat(k_dash,l)));
                            }
                        }
                    }
                }
            }
            // Combating numerical issues my substracting the max(log(k,))
            for(int k = 0; k < K; k++){
                float log_Tau_i_max=log_Tau_i.row(k).max();
                for(int k_dash = 0; k_dash < K; k_dash++){
                    //Tau_next(k,k_dash,i)=exp(log_Tau_i(k,k_dash));
                    Tau_next_t(k,k_dash,i)=exp(log_Tau_i(k,k_dash)-log_Tau_i_max); // Assigning the submatrix to original 3D array after taking exponential and substracting max
                }
            }
            Tau_next(increment_index,0)=Tau_next_t.slice(i);
            increment_index+=1;
        }
    }
    return Tau_next;
}

// [[Rcpp::export]]
vec grad_HMM_undir(vec theta, cube gamma, cube network, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    mat grad_vector(T_data,K);
    for (int t=0; t<T_data; t++){
        mat grad_mat(K,K,fill::zeros);
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat grad_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        grad_matsub(k,l)=gamma(i,k,t)*gamma(j,l,t)*(network(i,j,t)-(exp_val_mat(k,l)/(1+exp_val_mat(k,l))));
                    }
                }
                grad_mat+=grad_matsub;
            }
        }
        vec rsum=rowsum_Mat(grad_mat);
        vec csum=colsum_Mat(grad_mat);
        for(int k = 0; k < K; k++){
            grad_vector(t,k)=rsum(k)+csum(k);
        }
    }
    vec grad_vector_final(K);
    for(int k = 0; k < K; k++){
        grad_vector_final(k)=accu(grad_vector.col(k));
    }
    return grad_vector_final;
}

// [[Rcpp::export]]
mat hess_HMM_undir(vec theta, cube gamma, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    cube t1(K,K,T_data);
    mat hess_mat(K,K,fill::zeros);
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat hess_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        //float exp_val=exp(theta_u(k)+theta_u(l));
                        hess_matsub(k,l)=-(gamma(i,k,t)*gamma(j,l,t)*(exp_val_mat(k,l)/pow((1+exp_val_mat(k,l)),2)));
                    }
                }
                hess_mat+=hess_matsub;
            }
        }
        for(int k = 0; k < K; k++){
            for(int l = 0; l < K; l++){
                if(k!=l){
                    t1(k,l,t)=(hess_mat(k,l)+hess_mat(l,k));
                }
            }
        }
        vec rsum=rowsum_Mat(hess_mat);
        vec csum=colsum_Mat(hess_mat);
        for(int k = 0; k < K; k++){
            t1(k,k,t)=(csum(k)+rsum(k)+(2*hess_mat(k,k)));
        }
    }
    mat t2(K,K,fill::zeros);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            for (int t=0; t<T_data; t++){
                t2(k,l)+=t1(k,l,t);
            }
        }
    }
    return t2;
}

// [[Rcpp::export]]
float ELBO_conv_HMM_undir(cube gamma, vec alpha, cube pi, cube Tau, vec theta, cube network, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    
    float t1=0;
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        t1+=(gamma(i,k,t)*gamma(j,l,t)*(network(i,j,t)*(theta(k)+theta(l))-log(1+exp_val_mat(k,l))));
                    }
                }
            }
        }
    }
    
    float t2=0;
    for(int i = 0; i < N; i++){
        for(int k = 0; k < K; k++){
            if((alpha(k)>=(pow(10,(-100))))&(gamma(i,k,0)>=(pow(10,(-100))))){
                t2+=gamma(i,k,0)*(log(alpha(k))-log(gamma(i,k,0)));
            }
        }
    }
    
    float t3=0;
    for (int t=1; t<T_data; t++){
        for(int i = 0; i < N; i++){
            for(int k = 0; k < K; k++){
                for(int k_dash = 0; k_dash < K; k_dash++){
                    if((pi(k,k_dash,t-1)>=(pow(10,(-100))))&(Tau(k,k_dash,N*(t-1)+i)>=(pow(10,(-100))))){
                        t3+=gamma(i,k,t-1)*Tau(k,k_dash,N*(t-1)+i)*(log(pi(k,k_dash,t-1))-log(Tau(k,k_dash,N*(t-1)+i)));
                    }
                }
            }
        }
    }
    
    float ELBO_val=t1+t2+t3;
    return ELBO_val;
}


///////////////////////////////////////////////////////////////////////////////////////////////
// Defining functions for K=1

// [[Rcpp::export]]
float grad_HMM_undir_K1(float theta, cube network, int N, int T_data){
    vec grad_vector(T_data);
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(1+exp_val);
    for (int t=0; t<T_data; t++){
        float grad_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                grad_val_t+=network(i,j,t)-exp_val_ratio;
            }
        }
        grad_vector(t)=2*grad_val_t;
    }
    float grad_val=sum(grad_vector);
    return grad_val;
}

// [[Rcpp::export]]
float hess_HMM_undir_K1(float theta, int N, int T_data){
    vec hess_vector(T_data);
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(pow((1+exp_val),2));
    for (int t=0; t<T_data; t++){
        float hess_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                hess_val_t+=(-exp_val_ratio);
            }
        }
        hess_vector(t)=4*hess_val_t;
    }
    float hess_val=sum(hess_vector);
    return hess_val;
}

// [[Rcpp::export]]
float ELBO_conv_HMM_undir_K1(float theta, cube network, int N, int T_data){
    float exp_val=exp(2*theta);
    float log_exp_val=log(1+exp_val);
    float ELBO_val=0;
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                ELBO_val+=(network(i,j,t)*(2*theta)-log_exp_val);
            }
        }
    }
    return ELBO_val;
}
///////////////////////////////////////////////////////////////////////////////////////////////

