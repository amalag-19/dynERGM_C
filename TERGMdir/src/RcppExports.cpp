// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

// rowsum_Mat
vec rowsum_Mat(mat M);
RcppExport SEXP TERGMdir_rowsum_Mat(SEXP MSEXP) {
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
RcppExport SEXP TERGMdir_colsum_Mat(SEXP MSEXP) {
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
RcppExport SEXP TERGMdir_epan(SEXP inputSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type input(inputSEXP);
    rcpp_result_gen = Rcpp::wrap(epan(input));
    return rcpp_result_gen;
END_RCPP
}
// gamma_update_TERGM_dir
cube gamma_update_TERGM_dir(mat gamma, vec pi, mat theta, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMdir_gamma_update_TERGM_dir(SEXP gammaSEXP, SEXP piSEXP, SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< vec >::type pi(piSEXP);
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_update_TERGM_dir(gamma, pi, theta, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_dir_oe
vec grad_TERGM_dir_oe(mat theta, mat gamma, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMdir_grad_TERGM_dir_oe(SEXP thetaSEXP, SEXP gammaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_dir_oe(theta, gamma, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_dir_re
vec grad_TERGM_dir_re(mat theta, mat gamma, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMdir_grad_TERGM_dir_re(SEXP thetaSEXP, SEXP gammaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_dir_re(theta, gamma, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_oe
mat hess_TERGM_dir_oe(mat theta, mat gamma, int N, int K, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_oe(SEXP thetaSEXP, SEXP gammaSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_oe(theta, gamma, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_re
mat hess_TERGM_dir_re(mat theta, mat gamma, int N, int K, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_re(SEXP thetaSEXP, SEXP gammaSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_re(theta, gamma, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_oe_re
mat hess_TERGM_dir_oe_re(mat theta, mat gamma, int N, int K, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_oe_re(SEXP thetaSEXP, SEXP gammaSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_oe_re(theta, gamma, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// ELBO_conv_TERGM_dir
float ELBO_conv_TERGM_dir(mat gamma, vec pi, mat theta, cube network, int N, int K, int T_data);
RcppExport SEXP TERGMdir_ELBO_conv_TERGM_dir(SEXP gammaSEXP, SEXP piSEXP, SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP KSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< vec >::type pi(piSEXP);
    Rcpp::traits::input_parameter< mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(ELBO_conv_TERGM_dir(gamma, pi, theta, network, N, K, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_dir_K1_oe
float grad_TERGM_dir_K1_oe(vec theta, cube network, int N, int T_data);
RcppExport SEXP TERGMdir_grad_TERGM_dir_K1_oe(SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_dir_K1_oe(theta, network, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// grad_TERGM_dir_K1_re
float grad_TERGM_dir_K1_re(vec theta, cube network, int N, int T_data);
RcppExport SEXP TERGMdir_grad_TERGM_dir_K1_re(SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_TERGM_dir_K1_re(theta, network, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_K1_oe
float hess_TERGM_dir_K1_oe(vec theta, int N, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_K1_oe(SEXP thetaSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_K1_oe(theta, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_K1_re
float hess_TERGM_dir_K1_re(vec theta, int N, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_K1_re(SEXP thetaSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_K1_re(theta, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// hess_TERGM_dir_K1_oe_re
float hess_TERGM_dir_K1_oe_re(vec theta, int N, int T_data);
RcppExport SEXP TERGMdir_hess_TERGM_dir_K1_oe_re(SEXP thetaSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_TERGM_dir_K1_oe_re(theta, N, T_data));
    return rcpp_result_gen;
END_RCPP
}
// ELBO_conv_TERGM_dir_K1
float ELBO_conv_TERGM_dir_K1(vec theta, cube network, int N, int T_data);
RcppExport SEXP TERGMdir_ELBO_conv_TERGM_dir_K1(SEXP thetaSEXP, SEXP networkSEXP, SEXP NSEXP, SEXP T_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< cube >::type network(networkSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type T_data(T_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(ELBO_conv_TERGM_dir_K1(theta, network, N, T_data));
    return rcpp_result_gen;
END_RCPP
}