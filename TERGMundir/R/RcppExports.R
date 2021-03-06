# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

rowsum_Mat <- function(M) {
    .Call('TERGMundir_rowsum_Mat', PACKAGE = 'TERGMundir', M)
}

colsum_Mat <- function(M) {
    .Call('TERGMundir_colsum_Mat', PACKAGE = 'TERGMundir', M)
}

epan <- function(input) {
    .Call('TERGMundir_epan', PACKAGE = 'TERGMundir', input)
}

gamma_update_TERGM_undir <- function(gamma, alpha, theta, network, N, K, T_data) {
    .Call('TERGMundir_gamma_update_TERGM_undir', PACKAGE = 'TERGMundir', gamma, alpha, theta, network, N, K, T_data)
}

grad_TERGM_undir <- function(theta, gamma, network, N, K, T_data) {
    .Call('TERGMundir_grad_TERGM_undir', PACKAGE = 'TERGMundir', theta, gamma, network, N, K, T_data)
}

hess_TERGM_undir <- function(theta, gamma, N, K, T_data) {
    .Call('TERGMundir_hess_TERGM_undir', PACKAGE = 'TERGMundir', theta, gamma, N, K, T_data)
}

ELBO_conv_TERGM_undir <- function(gamma, alpha, theta, network, N, K, T_data) {
    .Call('TERGMundir_ELBO_conv_TERGM_undir', PACKAGE = 'TERGMundir', gamma, alpha, theta, network, N, K, T_data)
}

grad_TERGM_undir_K1 <- function(theta, network, N, T_data) {
    .Call('TERGMundir_grad_TERGM_undir_K1', PACKAGE = 'TERGMundir', theta, network, N, T_data)
}

hess_TERGM_undir_K1 <- function(theta, N, T_data) {
    .Call('TERGMundir_hess_TERGM_undir_K1', PACKAGE = 'TERGMundir', theta, N, T_data)
}

ELBO_conv_TERGM_undir_K1 <- function(theta, network, N, T_data) {
    .Call('TERGMundir_ELBO_conv_TERGM_undir_K1', PACKAGE = 'TERGMundir', theta, network, N, T_data)
}

