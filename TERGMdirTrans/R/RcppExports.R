# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

rowsum_Mat <- function(M) {
    .Call('TERGMdirTrans_rowsum_Mat', PACKAGE = 'TERGMdirTrans', M)
}

colsum_Mat <- function(M) {
    .Call('TERGMdirTrans_colsum_Mat', PACKAGE = 'TERGMdirTrans', M)
}

epan <- function(input) {
    .Call('TERGMdirTrans_epan', PACKAGE = 'TERGMdirTrans', input)
}

Trans_stat_cal <- function(network, N, T_data) {
    .Call('TERGMdirTrans_Trans_stat_cal', PACKAGE = 'TERGMdirTrans', network, N, T_data)
}

gamma_update_TERGM_dir_Trans <- function(gamma, pi, theta, network, Trans_stat, N, K, T_data) {
    .Call('TERGMdirTrans_gamma_update_TERGM_dir_Trans', PACKAGE = 'TERGMdirTrans', gamma, pi, theta, network, Trans_stat, N, K, T_data)
}

grad_TERGM_dir_Trans <- function(theta, gamma, network, Trans_stat, N, K, T_data) {
    .Call('TERGMdirTrans_grad_TERGM_dir_Trans', PACKAGE = 'TERGMdirTrans', theta, gamma, network, Trans_stat, N, K, T_data)
}

hess_TERGM_dir_Trans <- function(theta, gamma, N, K, T_data) {
    .Call('TERGMdirTrans_hess_TERGM_dir_Trans', PACKAGE = 'TERGMdirTrans', theta, gamma, N, K, T_data)
}

ELBO_conv_TERGM_dir_Trans <- function(gamma, alpha, theta, network, Trans_stat, N, K, T_data) {
    .Call('TERGMdirTrans_ELBO_conv_TERGM_dir_Trans', PACKAGE = 'TERGMdirTrans', gamma, alpha, theta, network, Trans_stat, N, K, T_data)
}

grad_TERGM_dir_K1_Trans <- function(theta, network, Trans_stat, N, T_data) {
    .Call('TERGMdirTrans_grad_TERGM_dir_K1_Trans', PACKAGE = 'TERGMdirTrans', theta, network, Trans_stat, N, T_data)
}

hess_TERGM_dir_K1_Trans <- function(theta, N, T_data) {
    .Call('TERGMdirTrans_hess_TERGM_dir_K1_Trans', PACKAGE = 'TERGMdirTrans', theta, N, T_data)
}

ELBO_conv_TERGM_dir_K1_Trans <- function(theta, network, Trans_stat, N, T_data) {
    .Call('TERGMdirTrans_ELBO_conv_TERGM_dir_K1_Trans', PACKAGE = 'TERGMdirTrans', theta, network, Trans_stat, N, T_data)
}

