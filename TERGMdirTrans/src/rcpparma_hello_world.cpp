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
// gamma update function, gradient and hessian functions and ELBO convergence function for TERGM

// [[Rcpp::export]]
cube Trans_stat_cal(cube network, int N, int T_data){
    cube Trans_stat(N,N,T_data-1,fill::zeros);
    for(int t = 0; t < (T_data-1); t++){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                if(j!=i){
                    float Trans_stat_num=0;
                    float Trans_stat_denom=0;
                    for (int trans_index = 0; trans_index < N; trans_index++){
                        if(trans_index!=i){
                            Trans_stat_num+=network(i,trans_index,t)*network(trans_index,j,t)*network(i,j,t+1);
                            Trans_stat_denom+=network(i,trans_index,t)*network(trans_index,j,t);
                        }
                    }
                    Trans_stat(i,j,t)=Trans_stat_num/Trans_stat_denom;
                }
            }
        }
    }
    return(Trans_stat);
}

// [[Rcpp::export]]
cube gamma_update_TERGM_dir_Trans(mat gamma, vec pi, vec theta, cube network, cube Trans_stat, int N, int K, int T_data){
    cube quad_lin_coeff(N,K,2);
    for(int i = 0; i < N; i++){
        if(i!=(N-1)){
            for(int k = 0; k < K; k++){
                float t1=0;
                for (int t=0; t<(T_data-1); t++){
                    for(int j = i+1; j < N; j++){
                        for(int l = 0; l < K; l++){
                            float exp_val=exp(theta(k)+theta(l));
                            t1+=(gamma(j,l)/(2*gamma(i,k)))*((Trans_stat(i,j,t)*(theta(k)+theta(l)))-log(1+exp_val));
                        }
                    }
                }
                quad_lin_coeff(i,k,0)=t1-(T_data/gamma(i,k));
                quad_lin_coeff(i,k,1)=T_data*(log(pi(k))-log(gamma(i,k))+1);
            }
        } else if(i==(N-1)){
            for(int k = 0; k < K; k++){
                quad_lin_coeff(i,k,0)=-(T_data/gamma((N-1),k));
                quad_lin_coeff(i,k,1)=T_data*(log(pi(k))-log(gamma((N-1),k))+1);
            }
        }
    }
    return quad_lin_coeff;
}

// [[Rcpp::export]]
mat grad_TERGM_dir_Trans(vec theta, mat gamma, cube network, cube Trans_stat, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    mat grad_vector(T_data-1,K);
    for (int t=0; t<(T_data-1); t++){
        mat grad_mat(K,K,fill::zeros);
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat grad_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        grad_matsub(k,l)=gamma(i,k)*gamma(j,l)*(Trans_stat(i,j,t)-(exp_val_mat(k,l)/(1+exp_val_mat(k,l))));
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
mat hess_TERGM_dir_Trans(vec theta, mat gamma, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    cube t1(K,K,T_data-1);
    mat hess_mat(K,K,fill::zeros);
    for (int t=0; t<(T_data-1); t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat hess_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        //float exp_val=exp(theta_u(k)+theta_u(l));
                        hess_matsub(k,l)=-(gamma(i,k)*gamma(j,l)*(exp_val_mat(k,l)/pow((1+exp_val_mat(k,l)),2)));
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
            for (int t=0; t<(T_data-1); t++){
                t2(k,l)+=t1(k,l,t);
            }
        }
    }
    return t2;
}


// [[Rcpp::export]]
float ELBO_conv_TERGM_dir_Trans(mat gamma, vec alpha, vec theta, cube network, cube Trans_stat, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    
    float t1=0;
    for (int t=0; t<(T_data-1); t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        t1+=(gamma(i,k)*gamma(j,l)*(Trans_stat(i,j,t)*(theta(k)+theta(l))-log(1+exp_val_mat(k,l))));
                    }
                }
            }
        }
    }
    
    float t2=0;
    for(int i = 0; i < N; i++){
        for(int k = 0; k < K; k++){
            if((alpha(k)>=(pow(10,(-100))))&(gamma(i,k)>=(pow(10,(-100))))){
                t2+=gamma(i,k)*(log(alpha(k))-log(gamma(i,k)));
            }
        }
    }

    float ELBO_val=t1+t2;
    return ELBO_val;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defining functions for directed, transitivity, K=1

// [[Rcpp::export]]
float grad_TERGM_dir_K1_Trans(float theta, cube network, cube Trans_stat, int N, int T_data){
    vec grad_vector(T_data-1);
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(1+exp_val);
    for (int t=0; t<(T_data-1); t++){
        float grad_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                grad_val_t+=(Trans_stat(i,j,t)-exp_val_ratio);
            }
        }
        grad_vector(t)=2*grad_val_t;
    }
    float grad_val=accu(grad_vector);
    return grad_val;
}

// [[Rcpp::export]]
float hess_TERGM_dir_K1_Trans(float theta, int N, int T_data){
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(pow((1+exp_val),2));
    vec hess_vector(T_data-1);
    for (int t=0; t<(T_data-1); t++){
        float hess_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                hess_val_t+=(-exp_val_ratio);
            }
        }
        hess_vector(t)=4*hess_val_t;
    }
    float hess_val=accu(hess_vector);
    return hess_val;
}

// [[Rcpp::export]]
float ELBO_conv_TERGM_dir_K1_Trans(float theta, cube network, cube Trans_stat, int N, int T_data){
    float exp_val=exp(2*theta);
    float log_exp_val=log(1+exp_val);
    
    float ELBO_val=0;
    for (int t=0; t<(T_data-1); t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                ELBO_val+=((Trans_stat(i,j,t)*2*theta)-log_exp_val);
            }
        }
    }
    return ELBO_val;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

