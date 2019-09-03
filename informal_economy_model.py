# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:33:47 2019

Informal economy model.
Implementation of Bento, Jacobsen and Liu, 2018
Paper title: 'Environmental Policy in the Presence of an Informal Sector'

The model is a static genral equilibrium, with a formal and an informal services
sector. The model builds the baseline calibrated for the USA. Then the equilibrium
is shocked with a tageted reduction in emissions (i.e. reduction in energy output)
achieved via a tax on energy. The script solves for the tax achieving that.

At the same time, government income is held fixed, so that the increased revenue 
from the energy tax is compensated by a reduction of the labour income tax that 
ensures fixed government income.

The effect on GDP of this policy reveals the potential effect of climate policy
(i.e. carbon taxes) on GDP and formal economy when accompanied with carbon revenue
recycling


"""
from scipy.optimize import fsolve, root
from types import SimpleNamespace
import numpy as np
import pickle
import pprint
import warnings

np.random.seed(0)

#
# DESCRIPTION OF ALL VARIABLES
#

#    E_m         :  Total energy produced
#    L_em        :  Total labour required in energy production
#    G           :  Manufacturing output
#    L_g         :  Labour used in manufacturing sector
#    E_g         :  Energy used in manufacturing sector
#    S_m         :  Formal services output
#    L_sm        :  Labour used in formal services
#    E_sm        :  Energy used in formal services
#    D_sm        :  Informal energy used in formal services
#    S_n         :  Informal services output
#    L_sn        :  Labour used in informal services
#    E_sn        :  Energy used in informal services
#    D_sn        :  Informal energy used in informal services
#    D           :  Informal energy output
#    L_dsm       :  Labour used to produced informal energy demand from formal services
#    L_dsn       :  Labour used to produced informal energy demand from informal services
#    pi          :  Profits of informal services firms
#    L           :  Total labour supplied
#    S           :  Total output of services (formal + informal)
#    p_L         :  Price of labour faced by formal firms (including labour tax, faced by manufacturinf, formal energy and formal services)
#    p_l         :  Value of leisure
#    w           :  Price of labour faced by informal firms (informal services and informal energy)
#    p_g         :  Price of manufacturing output
#    p_e         :  Price of formal energy
#    p_d         :  Price of informa energy
#    p_sm        :  Price of formal services
#    p_sn        :  Price of informal services
#    p_s         :  Price of services faced by consumers
#    B           :  Consumption aggregate in consumer utility function (including manufacturing and service consumption)
#    l           :  Leisure consumed by consumer
#    p_b         :  Price of aggregate consumptipon
#    p_u         :  Price of total utility
#    g           :  Government spending
#    B_spend     :  Total spending by conusmer on consumptoin of G and S (aggregated into B)
#    l_spend     :  Total spending by consumer on leisure
#    U           :  Utility of consumer
#    TI          :  Total income of consumer


# Calibrates elasticities in baseline


def calibrate(params):

    p = SimpleNamespace(**params)

    # Ratios of energy intensities
    ratio_M_SM  = p.intensity_G/p.intensity_SM    # X in the technical paper, manufacturing to services
    ratio_SF_SN = p.intensity_SM/p.intensity_SN   # Y in the technical paper, informal to formal

    # Quantities of labour from the excel shared by Anthony
    if(p.informal_share == 0):
        LinESN = 0
        LinESM = p.intensity_all / (1 + ratio_M_SM   *   p.G_share  /  p.S_share  )
        LinEG  = ratio_M_SM * LinESM * p.G_share / p.S_share
    else: 
        LinESN = p.intensity_all/(1+(ratio_M_SM*ratio_SF_SN*p.G_share/p.informal_share)+(ratio_SF_SN*p.S_share/p.informal_share))
        LinESM = LinESN*ratio_SF_SN*p.S_share/p.informal_share
        LinEG  = LinESN*ratio_M_SM*ratio_SF_SN*p.G_share/p.informal_share


    LinG = p.G_share-LinEG

    # Quantities of outputs from the excel shared by Anthony
    qdSM = p.S_share
    qdSN = p.informal_share
    qdE  = LinESN+LinESM+LinEG

    # Outputs
    G = p.G_share
    S_m = qdSM
    S_n = qdSN
    S = S_n+S_m
    L_em = qdE
    L_g = LinG
    if(p.informal_share == 0):
        E_sn = 0
    else:
        E_sn = L_em/(1+ratio_M_SM*ratio_SF_SN*G/S_n+ratio_SF_SN*S_m/S_n)
    E_sm = L_em/(S_n/(ratio_SF_SN*S_m)+ratio_M_SM*G/S_m+1)

    E_g = L_em/(S_n/(ratio_M_SM*ratio_SF_SN*G)+1+S_m/(ratio_M_SM*G))
    I_g = E_g / G

    # Constants
    p_l = 1
    w   = p_l
    D_sm =0 # As discussed with anthony
    gamma_g =1
    p_d=1

    # prices

    p_L = p_l + p.tau_L
    p_e = 1 + p.tau_L + p.tau_E
    p_n = p_e  # price of informal energy is the same as the price of formal energy

    theta_EM = p_e*E_sm/(p_n * S_m)
    theta_DM = 0
    theta_LM = 1 - theta_EM - theta_DM

    if(p.informal_share == 0):
        theta_EN = 0
        theta_DN = 0
        theta_LN = 1
        L_sn = 0
    else:
        theta_EN = p_e*E_sn/(p_n*S_n)
        theta_DN = (p_d / p_e)  * (p.D_sn / E_sn) * theta_EN # by definition
        theta_LN = 0.4 - theta_EN - theta_DN
        L_sn = ((E_sn * theta_LN * p_e) / (theta_EN * w))

    p_s  = (p_e * E_sm) / (theta_EM * S_m)
    L_sm = (p_s*theta_LM*S_m)/p_L

    # alphas in manufacturing. #CHANGED from w to p_L -- check with Anthony...

    alpha_lg = ((p_L/p_e)**p.sigma_g*L_g/E_g)/(1+(p_L/p_e)**p.sigma_g*L_g/E_g)
    alpha_eg = ((p_e/p_L)**p.sigma_g*E_g/L_g)/(1+(p_e/p_L)**p.sigma_g*E_g/L_g)

    p_g = (alpha_lg*p_L**(1-p.sigma_g)+alpha_eg*p_e**(1-p.sigma_g))**(1/(1-p.sigma_g))

    if(p.informal_share == 0):
        gamma_sn = 0
    else:
        gamma_sn = S_n/(L_sn**theta_LN*E_sn**theta_EN*p.D_sn**theta_DN)

    gamma_sm = S_m/(L_sm**theta_LM*E_sm**theta_EM) # NOTE got rid of D_sm

    alpha_bg = 1/(1 + (p_s / p_g)*(S/G)**(1/p.sigma_b))
    alpha_bs = 1 - alpha_bg
     
    B = (alpha_bg*G**((p.sigma_b-1)/p.sigma_b)+alpha_bs*S**((p.sigma_b-1)/p.sigma_b))**(p.sigma_b/(p.sigma_b-1))
    p_b = (alpha_bg**p.sigma_b*p_g**(1-p.sigma_b)+alpha_bs**p.sigma_b*p_s**(1-p.sigma_b))**(1/(1-p.sigma_b))

    L_tot = (L_g+L_sn+L_sm+L_em+p.D_sn)*p.Lratio
    l = L_tot-(L_g+L_sn+L_sm+L_em+p.D_sn)

    # TODO CHECK THESE, MISTAKE in Anthony's stuff somewhere

    alpha_ub = (B**(1/p.sigma_u)*p_b)  / (B**(1/p.sigma_u)*p_b + l**(1/p.sigma_u)*p_l)
    alpha_ul = 1 - alpha_ub

    p_u = (alpha_ul**p.sigma_u*p_l**(1-p.sigma_u)+alpha_ub**p.sigma_u*p_b**(1-p.sigma_u))**(1/(1-p.sigma_u))

    pi = (p_s*S_n-w*L_sn-p_e*E_sn-p_d*p.D_sn)
    L = L_g+E_g+L_sm+E_sm+L_sn+E_sn+p.D_sn #L_dsm = D_sm and L_dsn = D_sn
    S = S_n + S_m
    g = p.tau_L*(L_tot-L_sn-p.D_sn-l)-p.tau_E*L_em # E_m = L_em
    TI = L_tot*w+g+pi
    B_spend = TI*p_b*(w*alpha_ub)**p.sigma_u/(p_b*(alpha_ub*w)**p.sigma_u+w*(p_b*alpha_ul)**p.sigma_u)
    l_spend = TI-B_spend


    # Check that some equalities hold
    s = np.sum(np.absolute(np.array([ p_g*G-(w+p.tau_L+p.tau_E)*E_g-(w+p.tau_L)*L_g,
                p_s*S_m-(w+p.tau_L+p.tau_E)*E_sm-(w+p.tau_L)*L_sm,
                w*(L_g+E_g+L_sm+E_sm+L_sn+E_sn+p.D_sn) + p.tau_L*(L_g+E_g+L_sm+E_sm+E_sn) + p.tau_E*(E_g+E_sm+E_sn)+(p_s*S_n-(w*L_sn+p_e*E_sn+p_d*p.D_sn))-p_g*G-p_s*(S_n+S_m),
                alpha_bg/alpha_bs*(G/S)**(-1/p.sigma_b)-p_g/p_s,
                alpha_ub/alpha_ul*(l/B)**(1/p.sigma_u)-p_b/p_l
        ])))

    assert(s < 0.001)


    # Check that the economy is in equilibrium

    assert(np.sum(np.absolute(np.array([ #production

              G-gamma_g*(alpha_lg**(1/p.sigma_g)*L_g**((p.sigma_g-1)/p.sigma_g)+alpha_eg**(1/p.sigma_g)*E_g**((p.sigma_g-1)/p.sigma_g))**(p.sigma_g/(p.sigma_g-1)), #PROBLEMATIC
              S_m-gamma_sm*L_sm**theta_LM*E_sm**theta_EM,
              S_n-gamma_sn*L_sn**theta_LN*E_sn**theta_EN*p.D_sn**theta_DN,
              L-L_g-E_g-L_sm-E_sm-L_sn-E_sn-D_sm-p.D_sn,
              # manufacturing
              L_g-alpha_lg*(p_g/p_L)**p.sigma_g*G, #MW:Because c_g = p_g # PA: correct
              E_g-alpha_eg*(p_g/p_e)**p.sigma_g*G, #MW:Because c_g = p_g # PA: correct
              # formal services
              L_sm-(p_s*theta_LM*S_m)/p_L, 
              E_sm-(theta_EM/theta_LM)*(p_L/p_e)*L_sm, 
              # informal services
              L_sn-(p_s*theta_LN*S_n)/w, #w because informal firm , hence does not pay tax on labour
              E_sn-(theta_EN/theta_LN)*(w/p_e)*L_sn,
              p.D_sn-(theta_DN/theta_LN)*(w/p_d)*L_sn,  
              pi-(p_s*S_n-w*L_sn-p_e*E_sn-p_d*p.D_sn),
              # prices
              p_g-(alpha_lg*p_L**(1-p.sigma_g)+alpha_eg*p_e**(1-p.sigma_g))**(1/(1-p.sigma_g)),
              p_s-(1/gamma_sm)*(p_L/theta_LM)**theta_LM*(p_e/theta_EM)**theta_EM,
              # Utility and demand
              B-(alpha_bg*G**((p.sigma_b-1)/p.sigma_b)+alpha_bs*S**((p.sigma_b-1)/p.sigma_b))**(p.sigma_b/(p.sigma_b-1)),
              p_b-(alpha_bg**p.sigma_b*p_g**(1-p.sigma_b)+alpha_bs**p.sigma_b*p_s**(1-p.sigma_b))**(1/(1-p.sigma_b)),
              p_u-(alpha_ul**p.sigma_u*p_l**(1-p.sigma_u)+alpha_ub**p.sigma_u*p_b**(1-p.sigma_u))**(1/(1-p.sigma_u)), 
              TI-L_tot*w-g-pi, 
              B_spend-TI*p_b*(w*alpha_ub)**p.sigma_u/(p_b*(alpha_ub*w)**p.sigma_u+w*(p_b*alpha_ul)**p.sigma_u),
              l_spend-TI*w*(p_b*alpha_ul)**p.sigma_u/(p_b*(alpha_ub*w)**p.sigma_u+w*(p_b*alpha_ul)**p.sigma_u),
              l_spend-TI+B_spend, 
              G-(alpha_bg/p_g)**p.sigma_b*(B_spend)/(alpha_bg**p.sigma_b*p_g**(1-p.sigma_b)+alpha_bs**p.sigma_b*p_s**(1-p.sigma_b)), #PROBLEM
              S-(alpha_bs/p_s)**p.sigma_b*(B_spend)/(alpha_bg**p.sigma_b*p_g**(1-p.sigma_b)+alpha_bs**p.sigma_b*p_s**(1-p.sigma_b)), #PROBLEM
              S-S_m-S_n,
              # Government
              g-p.tau_L*(L_tot-L_sn-p.D_sn-l)-p.tau_E*(E_g+E_sm+E_sn), 
              ## Add ons for unique sol
              l+L-L_tot
              ]))) < 0.001)


    calibration = {
      'gamma_g': gamma_g,
      'alpha_lg': alpha_lg,
      'alpha_eg': alpha_eg,
      'gamma_sm': gamma_sm,
      'theta_LM': theta_LM,
      'theta_EM': theta_EM,
      'theta_DM': theta_DM,
      'gamma_sn': gamma_sn,
      'theta_LN': theta_LN,
      'theta_EN': theta_EN,
      'theta_DN': theta_DN,
      'alpha_ub': alpha_ub,
      'alpha_ul': alpha_ul,
      'alpha_bg': alpha_bg,
      'alpha_bs': alpha_bs,
      'L_tot': L_tot,
      'sigma_g': p.sigma_g,
      'sigma_b': p.sigma_b,
      'sigma_u': p.sigma_u,
      'tau_L': p.tau_L,
      'tau_E': p.tau_E,

    }

    baseline = [G, L_g, E_g, S_m, \
    L_sm, E_sm, S_n, \
    L_sn, E_sn, p.D_sn, \
    L,  p_g, p_e, \
    p_s, S_n, pi,  \
    p_d, B, p_b, p_u, \
    TI, L_tot, l, g, \
    B_spend, l_spend, p.tau_L, p.tau_E]

    return((calibration, baseline))


def model_equations(variables):

    # Constants
    p_l = 1
    w = p_l
    gamma_g = 1

    G, L_g, E_g, S_m, \
    L_sm, E_sm, S_n, \
    L_sn, E_sn, D_sn, \
    L,  p_g, p_e, \
    p_s, S_n, pi,  \
    p_d, B, p_b, p_u, \
    TI, L_tot, l, g, \
    B_spend, l_spend, tau_L, tau_E = variables

    return( # production
              G-gamma_g*(alpha_lg**(1/sigma_g)*L_g**((sigma_g-1)/sigma_g)+alpha_eg**(1/sigma_g)*E_g**((sigma_g-1)/sigma_g))**(sigma_g/(sigma_g-1)), #PROBLEMATIC
              S_m-gamma_sm*L_sm**theta_LM*E_sm**theta_EM,
              S_n-gamma_sn*L_sn**theta_LN*E_sn**theta_EN*D_sn**theta_DN,
              L-L_g-E_g-L_sm-E_sm-L_sn-E_sn-D_sn,
              # manufacturing
              L_g-alpha_lg*(p_g/(p_l + tau_L))**sigma_g*G,  # c_g = p_g
              E_g-alpha_eg*(p_g/p_e)**sigma_g*G,  # c_g = p_g
              # formal services
              L_sm-(p_s*theta_LM*S_m)/(p_l + tau_L),
              E_sm-(theta_EM/theta_LM)*((p_l + tau_L)/p_e)*L_sm,
              # informal services
              L_sn-(p_s*theta_LN*S_n)/w,
              E_sn-(theta_EN/theta_LN)*(w/p_e)*L_sn,
              D_sn-(theta_DN/theta_LN)*(w/p_d)*L_sn,
              pi-(p_s*S_n-w*L_sn-p_e*E_sn-p_d*D_sn),
              # prices
              p_e-1-tau_E-tau_L,
              p_g-(alpha_lg*(p_l + tau_L)**(1-sigma_g)+alpha_eg*p_e**(1-sigma_g))**(1/(1-sigma_g)),
              p_s-(1/gamma_sm)*((p_l + tau_L)/theta_LM)**theta_LM*(p_e/theta_EM)**theta_EM,
              # Utility and demand
              p_b-(alpha_bg**sigma_b*p_g**(1-sigma_b)+alpha_bs**sigma_b*p_s**(1-sigma_b))**(1/(1-sigma_b)),
              p_u-(alpha_ul**sigma_u*p_l**(1-sigma_u)+alpha_ub**sigma_u*p_b**(1-sigma_u))**(1/(1-sigma_u)), 
              TI-L_tot*w-g-pi,
              B_spend-TI*p_b*(w*alpha_ub)**sigma_u/(p_b*(alpha_ub*w)**sigma_u+w*(p_b*alpha_ul)**sigma_u),
              l_spend-TI*w*(p_b*alpha_ul)**sigma_u/(p_b*(alpha_ub*w)**sigma_u+w*(p_b*alpha_ul)**sigma_u),
              l_spend-TI+B_spend,
              G-(alpha_bg/p_g)**sigma_b*(B_spend)/(alpha_bg**sigma_b*p_g**(1-sigma_b)+alpha_bs**sigma_b*p_s**(1-sigma_b)), #PROBLEM
              S_n+S_m-(alpha_bs/p_s)**sigma_b*(B_spend)/(alpha_bg**sigma_b*p_g**(1-sigma_b)+alpha_bs**sigma_b*p_s**(1-sigma_b)), #PROBLEM
              # Government
              g-tau_L*(L_tot-L_sn-D_sn-l)-tau_E*(E_g+E_sm+E_sn),
              l+L-L_tot,
              p_d-1,
              E_g+E_sm+E_sn - TARGET_E_M,
              g-TARGET_g
            )

# params for the baseline, provided in the paper
params = {

  'Lratio': 1.6,         # ratio of total labour to labour supply (assumed constant all countries)
  'Dratio': 1,            # Not used

  'tau_L': 0.4,  # tax on labour
  'tau_E': 0,  # tax on energy/emissions

  'sigma_g': 1.01,        # elasticity of subs between L and E in manufacturing (assumed constant all countries)

  # Utility function
  'sigma_u': 0.9,         # elasticity of substitution between leisure and consumption (assumed constant all countries)
  'sigma_b': 1.01,        # elasticity of substitution between manufacturing and services (assumed constant all countries)

  # Energy intensity ratios
  'intensity_all': 4.1,    # energy intensity of the whole economy (economy_energy_intensity)
  'intensity_G': 8.2,      # energy intensity of the manufacturing sector  (manuf_energy_intensity)

  'intensity_SM': 2.6,     # energy intensity of the formal service sector (services_energy_intensity)
  'intensity_SN': 2.6,     # energy intensity of the informal service sector (=services_energy_intensity)


  # Composition of the economy
  'G_share': 22.1,        # manufacture share of output, from Spreadsheet
  'S_share': 77.9,        # Formal services share of output, from Spreadsheet
  'informal_share': 8.4,  # Size of the informal economy, from Spreadsheet

  'D_sn': 0  # TODO should be calibrating D, not D_sn (but here have got no DSM at the moment). TODO use GAINS outputs for D.

}

#
# Initial calibration -- returns parameters and equilibrium values
#
calibration_params, calibration_conditions = calibrate(params)

G, L_g, E_g, S_m, \
    L_sm, E_sm, S_n, \
    L_sn, E_sn, D_sn, \
    L,  p_g, p_e, \
    p_s, S_n, pi,  \
    p_d, B, p_b, p_u, \
    TI, L_tot, l, g, \
    B_spend, l_spend, tau_L, tau_E = calibration_conditions

globals().update(calibration_params)


# Get baseline energy production and government expenditure
baseline_E_M = E_g + E_sm + E_sn
baseline_g = g

#
# Check that the baseline model solves
#

# Set target energy production and government expenditure to be same as in baseline 
# and check that solution is same as that coming out of calibration

TARGET_E_M = baseline_E_M
TARGET_g = baseline_g

warnings.filterwarnings('ignore')

baseline_roots = root(model_equations, calibration_conditions, method='broyden1')
assert(baseline_roots.success)
assert(np.array_equal(baseline_roots.x, calibration_conditions))


print(f"GDP before tax: {G + S_m}")

#
# Set a target E_M and g and solve
#

TARGET_E_M = 0.9*baseline_E_M
TARGET_g = baseline_g

a = root(model_equations, calibration_conditions)
assert(a.success)

G, L_g, E_g, S_m, \
    L_sm, E_sm, S_n, \
    L_sn, E_sn, D_sn, \
    L,  p_g, p_e, \
    p_s, S_n, pi,  \
    p_d, B, p_b, p_u, \
    TI, L_tot, l, g, \
    B_spend, l_spend, tau_L, tau_E = a.x

print(f"GDP after tax: {G + S_m}")
