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
mat gamma_init_update_HMM_dir(mat gamma_init, vec log_alpha, mat theta, mat network_init, int N, int K){
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
                    float exp_val_1=exp(theta(k,0));
                    float exp_val_2=exp(theta(l,0));
                    float exp_val_3=exp(theta(k,1)+theta(l,1));
                    int indicator_10=(network_init(i,j)==1)&(network_init(j,i)==0);
                    int indicator_01=(network_init(i,j)==0)&(network_init(j,i)==1);
                    int indicator_11=(network_init(i,j)==1)&(network_init(j,i)==1);
                    log_gamma_init_i(k)+=gamma_init(j,l)*((indicator_10*theta(k,0))+(indicator_01*theta(l,0))+(indicator_11*(theta(k,1)+theta(l,1)))-log(1+exp_val_1+exp_val_2+exp_val_3));
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
cube gamma_update_HMM_dir(mat gamma_init, cube Tau, int N, int K, int T_data){
    cube gamma_next(N,K,T_data,fill::zeros);
    gamma_next.slice(0)=gamma_init;
    for(int t = 1; t < T_data; t++){
        for(int i = 0; i < N; i++){
            for(int k = 0; k < K; k++){
                for(int k_dash = 0; k_dash < K; k_dash++){
                    gamma_next(i,k,t)+=(gamma_next(i,k_dash,t-1)*Tau(k_dash,k,N*(t-1)+i));
                }
                // Adding small term to make numerically stable
                // gamma_next(i,k,t)+=pow(10,(-200));
            }
            for(int l = 0; l < K; l++){
                if(accu((gamma_next.slice(t)).col(l))<(pow(10,(-5)))){
                    gamma_next(0,l,t)+=pow(10,(-10));
                    // Renormalizing
                    float row_sum=accu((gamma_next.slice(t)).row(0));
                    for (int m = 0; m < K; m++){
                        gamma_next(0,m,t)=gamma_next(0,m,t)/row_sum;
                    }
                }
            }
        }
    }
    return gamma_next;
}

// [[Rcpp::export]]
field<mat> Tau_update_HMM_dir(cube gamma, cube log_pi, mat theta, cube network, int N, int K, int T_data){
    field<mat> Tau_next(N*(T_data-1),1);
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
                                float exp_val_1=exp(theta(k_dash,0));
                                float exp_val_2=exp(theta(l,0));
                                float exp_val_3=exp(theta(k_dash,1)+theta(l,1));
                                vec alpha(4,fill::zeros);
                                alpha(1)=exp_val_1;
                                alpha(2)=exp_val_2;
                                alpha(3)=exp_val_3;
                                float alpha_max=alpha.max();
                                float exp_val_1_mod=exp(theta(k_dash,0)-alpha_max);
                                float exp_val_2_mod=exp(theta(l,0)-alpha_max);
                                float exp_val_3_mod=exp(theta(k_dash,1)+theta(l,1)-alpha_max);
                                float log_exp_val=alpha_max+log(exp(-alpha_max)+exp_val_1_mod+exp_val_2_mod+exp_val_3_mod);
                                int indicator_10=(network(i,j,t+1)==1)&(network(j,i,t+1)==0);
                                int indicator_01=(network(i,j,t+1)==0)&(network(j,i,t+1)==1);
                                int indicator_11=(network(i,j,t+1)==1)&(network(j,i,t+1)==1);
                                log_Tau_i(k,k_dash)+=gamma(j,l,t)*((indicator_10*theta(k_dash,0))+(indicator_01*theta(l,0))+((indicator_11)*(theta(k_dash,1)+theta(l,1)))-log_exp_val);
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
mat grad_HMM_dir_oe(mat theta, cube gamma, cube network, int N, int K, int T_data){
    mat exp_val_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            exp_val_mat(k,l)=exp(theta(k)+theta(l));
        }
    }
    mat grad_vector(T_data,K);
    for (int t=0; t<T_data; t++){
        mat grad_mat_1(K,K,fill::zeros);
        mat grad_mat_2(K,K,fill::zeros);
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat grad_matsub_1(K,K);
                mat grad_matsub_2(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        int indicator_10=(network(i,j,t)==1)&(network(j,i,t)==0);
                        int indicator_01=(network(i,j,t)==0)&(network(j,i,t)==1);
                        grad_matsub_1(k,l)=gamma(i,k,t)*gamma(j,l,t)*(indicator_10-(exp_val_1/(1+exp_val_1+exp_val_2+exp_val_3)));
                        grad_matsub_2(k,l)=gamma(i,k,t)*gamma(j,l,t)*(indicator_01-(exp_val_2/(1+exp_val_1+exp_val_2+exp_val_3)));
                    }
                }
                grad_mat_1+=grad_matsub_1;
                grad_mat_2+=grad_matsub_2;
            }
        }
        vec rsum_mat_1=rowsum_Mat(grad_mat_1);
        vec csum_mat_2=colsum_Mat(grad_mat_2);
        for(int k = 0; k < K; k++){
            grad_vector(t,k)=rsum_mat_1(k)+csum_mat_2(k);
        }
    }
    vec grad_vector_final(K);
    for(int k = 0; k < K; k++){
        grad_vector_final(k)=accu(grad_vector.col(k));
    }
    return grad_vector_final;
}

// [[Rcpp::export]]
mat grad_HMM_dir_re(mat theta, cube gamma, cube network, int N, int K, int T_data){
    mat grad_vector(T_data,K);
    for (int t=0; t<T_data; t++){
        mat grad_mat(K,K,fill::zeros);
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat grad_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        int indicator_11=(network(i,j,t)==1)&(network(j,i,t)==1);
                        grad_matsub(k,l)=gamma(i,k,t)*gamma(j,l,t)*(indicator_11-(exp_val_3/(1+exp_val_1+exp_val_2+exp_val_3)));
                    }
                }
                grad_mat+=grad_matsub;
            }
        }
        vec rsum_mat=rowsum_Mat(grad_mat);
        vec csum_mat=colsum_Mat(grad_mat);
        for(int k = 0; k < K; k++){
            grad_vector(t,k)=rsum_mat(k)+csum_mat(k);
        }
    }
    vec grad_vector_final(K);
    for(int k = 0; k < K; k++){
        grad_vector_final(k)=accu(grad_vector.col(k));
    }
    return grad_vector_final;
}


// [[Rcpp::export]]
mat hess_HMM_dir_oe(mat theta, cube gamma, int N, int K, int T_data){
    cube t1(K,K,T_data);
    mat hess_mat_1(K,K,fill::zeros);
    mat hess_mat_2(K,K,fill::zeros);
    mat hess_mat_3(K,K,fill::zeros);
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat hess_matsub_1(K,K);
                mat hess_matsub_2(K,K);
                mat hess_matsub_3(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        hess_matsub_1(k,l)=(gamma(i,k,t)*gamma(j,l,t)*((exp_val_1*exp_val_2)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
                        hess_matsub_2(k,l)=(gamma(i,k,t)*gamma(j,l,t)*((exp_val_1+exp_val_1*exp_val_3)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
                        hess_matsub_3(k,l)=(gamma(i,k,t)*gamma(j,l,t)*((exp_val_2+exp_val_2*exp_val_3)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
                    }
                }
                hess_mat_1+=hess_matsub_1;
                hess_mat_2+=hess_matsub_2;
                hess_mat_3+=hess_matsub_3;
            }
        }
        for(int k = 0; k < K; k++){
            for(int l = 0; l < K; l++){
                if(k!=l){
                    t1(k,l,t)=(hess_mat_1(k,l)+hess_mat_2(l,k));
                }
            }
        }
        vec rsum_1=rowsum_Mat(hess_mat_1);
        vec csum_1=colsum_Mat(hess_mat_1);
        vec rsum_2=rowsum_Mat(hess_mat_2);
        vec csum_3=colsum_Mat(hess_mat_3);
        for(int k = 0; k < K; k++){
            t1(k,k,t)=(-csum_3(k)-rsum_2(k)-(rsum_1(k)+csum_1(k)-2*hess_mat_1(k,k)));
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
mat hess_HMM_dir_re(mat theta, cube gamma, int N, int K, int T_data){
    cube t1(K,K,T_data);
    mat hess_mat(K,K,fill::zeros);
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat hess_matsub(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        hess_matsub(k,l)=(gamma(i,k,t)*gamma(j,l,t)*(((1+exp_val_1+exp_val_2)*exp_val_3)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
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
            t1(k,k,t)=(-csum(k)-rsum(k)-2*hess_mat(k,k));
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
mat hess_HMM_dir_oe_re(mat theta, cube gamma, int N, int K, int T_data){
    cube t1(K,K,T_data);
    mat hess_mat_1(K,K,fill::zeros);
    mat hess_mat_2(K,K,fill::zeros);
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                mat hess_matsub_1(K,K);
                mat hess_matsub_2(K,K);
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        hess_matsub_1(k,l)=(gamma(i,k,t)*gamma(j,l,t)*((exp_val_1*exp_val_3)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
                        hess_matsub_2(k,l)=(gamma(i,k,t)*gamma(j,l,t)*((exp_val_2*exp_val_3)/(pow((1+exp_val_1+exp_val_2+exp_val_3),2))));
                    }
                }
                hess_mat_1+=hess_matsub_1;
                hess_mat_2+=hess_matsub_2;
            }
        }
        for(int k = 0; k < K; k++){
            for(int l = 0; l < K; l++){
                if(k!=l){
                    t1(k,l,t)=(hess_mat_1(k,l)+hess_mat_2(l,k));
                }
            }
        }
        vec rsum_1=rowsum_Mat(hess_mat_1);
        vec csum_2=colsum_Mat(hess_mat_2);
        for(int k = 0; k < K; k++){
            t1(k,k,t)=(csum_2(k)+rsum_1(k)+(hess_mat_1(k,k)+hess_mat_2(k,k)));
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
float ELBO_conv_HMM_dir(cube gamma, vec alpha, cube pi, cube Tau, mat theta, cube network, int N, int K, int T_data){
    float t1=0;
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        float exp_val_1=exp(theta(k,0));
                        float exp_val_2=exp(theta(l,0));
                        float exp_val_3=exp(theta(k,1)+theta(l,1));
                        int indicator_10=(network(i,j,t)==1)&(network(j,i,t)==0);
                        int indicator_01=(network(i,j,t)==0)&(network(j,i,t)==1);
                        int indicator_11=(network(i,j,t)==1)&(network(j,i,t)==1);
                        t1+=(gamma(i,k,t)*gamma(j,l,t)*((indicator_10*theta(k,0))+(indicator_01*theta(l,0))+(indicator_11*(theta(k,1)+theta(l,1)))-log(1+exp_val_1+exp_val_2+exp_val_3)));
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


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Defining functions for K=1

// [[Rcpp::export]]
float grad_HMM_dir_oe_K1(vec theta, cube network, int N, int T_data){
    vec grad_vector(T_data);
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float exp_val_ratio=(2*exp_val_1)/(1+2*exp_val_1+exp_val_2);
    for (int t=0; t<T_data; t++){
        float grad_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                int indicator_10=(network(i,j,t)==1)&(network(j,i,t)==0);
                int indicator_01=(network(i,j,t)==0)&(network(j,i,t)==1);
                grad_val_t+=((indicator_10+indicator_01)-exp_val_ratio);
            }
        }
        grad_vector(t)=grad_val_t;
    }
    float grad_val_oe=accu(grad_vector);
    return grad_val_oe;
}

// [[Rcpp::export]]
float grad_HMM_dir_re_K1(vec theta, cube network, int N, int T_data){
    vec grad_vector(T_data);
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float exp_val_ratio=(2*exp_val_2)/(1+2*exp_val_1+exp_val_2);
    for (int t=0; t<T_data; t++){
        float grad_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                int indicator_11=(network(i,j,t)==1)&(network(j,i,t)==1);
                grad_val_t+=((2*indicator_11)-exp_val_ratio);
            }
        }
        grad_vector(t)=grad_val_t;
    }
    float grad_val_re=accu(grad_vector);
    return grad_val_re;
}

// [[Rcpp::export]]
float hess_HMM_dir_oe_K1(vec theta, int N, int T_data){
    vec hess_vector(T_data);
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float exp_val_ratio=(exp_val_1+exp_val_1*exp_val_2)/(pow((1+2*exp_val_1+exp_val_2),2));
    for (int t=0; t<T_data; t++){
        float hess_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                hess_val_t+=exp_val_ratio;
            }
        }
        hess_vector(t)=-2*hess_val_t;
    }
    float hess_val_oe=accu(hess_vector);
    return hess_val_oe;
}

// [[Rcpp::export]]
float hess_HMM_dir_re_K1(vec theta, int N, int T_data){
    vec hess_vector(T_data);
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float Num=4*(exp_val_2+(2*exp_val_1*exp_val_2));
    float Denom=pow((1+2*exp_val_1+exp_val_2),2);
    float exp_val_ratio=Num/Denom;
    for (int t=0; t<T_data; t++){
        float hess_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                hess_val_t+=(-exp_val_ratio);
            }
        }
        hess_vector(t)=hess_val_t;
    }
    float hess_val_re=accu(hess_vector);
    return hess_val_re;
}

// [[Rcpp::export]]
float hess_HMM_dir_oe_re_K1(vec theta, int N, int T_data){
    vec hess_vector(T_data);
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float exp_val_ratio=(4*exp_val_1*exp_val_2)/(pow((1+2*exp_val_1+exp_val_2),2));
    for (int t=0; t<T_data; t++){
        float hess_val_t=0;
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                hess_val_t+=exp_val_ratio;
            }
        }
        hess_vector(t)=hess_val_t;
    }
    float hess_val=accu(hess_vector);
    return hess_val;
}

// [[Rcpp::export]]
float ELBO_conv_HMM_dir_K1(vec theta, cube network, int N, int T_data){
    float exp_val_1=exp(theta(0));
    float exp_val_2=exp(2*theta(1));
    float log_exp_val=log(1+2*exp_val_1+exp_val_2);
    float ELBO_val=0;
    for (int t=0; t<T_data; t++){
        for(int i = 0; i < (N-1); i++){
            for(int j = i+1; j < N; j++){
                int indicator_10=(network(i,j,t)==1)&(network(j,i,t)==0);
                int indicator_01=(network(i,j,t)==0)&(network(j,i,t)==1);
                int indicator_11=(network(i,j,t)==1)&(network(j,i,t)==1);
                ELBO_val+=((indicator_10*theta(0))+(indicator_01*theta(0))+(indicator_11*(2*theta(1)))-log_exp_val);
            }
        }
    }
    return ELBO_val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
