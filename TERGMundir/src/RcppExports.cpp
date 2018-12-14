// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

// rowsum_Mat
vec rowsum_Mat(mat M);
RcppExport SEXP TERGMundir_rowsum_Mat(SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(rowsum_Mat(M));
    return rcpp_result_gen;
END_RCPP
}
// colsum_Mat
vec colsum_Mat(mat M);
RcppExport SEXP TERGMundir_colsum_Mat(SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(colsum_Mat(M));
    return rcpp_result_gen;
END_RCPP
}
// epan
float epan(float input);
RcppExport SEXP TERGMundir_epan(SEXP inputSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type input(inputSEXP);
    rcpp_result_gen = Rcpp::wrap(epan(input));
    return rcpp_result_gen;
END_RCPP
}
// gamma_update_TERGM_undir
cube gamma_update_TERGM_undir(mat gamma, vec alpha, vec theta, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMundir_gamma_update_TERGM_undir(SEXP gammaSEXP, SEXP alphaSEXP, SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_update_TERGM_undir(gamma, alpha, theta, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_undir
mat grad_TERGM_undir(vec theta, mat gamma, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMundir_grad_TERGM_undir(SEXP thetaSEXP, SEXP gammaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_undir(theta, gamma, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_undir
mat hess_TERGM_undir(vec theta, mat gamma, int N, int K, int T_data);
RcppExport SEXP TERGMundir_hess_TERGM_undir(SEXP thetaSEXP, SEXP gammaSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_undir(theta, gamma, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// ELBO_conv_TERGM_undir
float ELBO_conv_TERGM_undir(mat gamma, vec alpha, vec theta, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMundir_ELBO_conv_TERGM_undir(SEXP gammaSEXP, SEXP alphaSEXP, SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(ELBO_conv_TERGM_undir(gamma, alpha, theta, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_undir_K1
float grad_TERGM_undir_K1(float theta, cube network, int N, int T_data);
RcppExport SEXP TERGMundir_grad_TERGM_undir_K1(SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_undir_K1(theta, network, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_undir_K1
float hess_TERGM_undir_K1(float theta, int N, int T_data);
RcppExport SEXP TERGMundir_hess_TERGM_undir_K1(SEXP thetaSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_undir_K1(theta, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// ELBO_conv_TERGM_undir_K1
float ELBO_conv_TERGM_undir_K1(float theta, cube network, int N, int T_data);
RcppExport SEXP TERGMundir_ELBO_conv_TERGM_undir_K1(SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(ELBO_conv_TERGM_undir_K1(theta, network, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
