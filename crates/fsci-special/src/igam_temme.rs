//! The regularized incomplete gamma functions `P(a, x)`, `Q(a, x)` and their inverses as SciPy
//! 1.17.1 computes them, ported from xsf 0d0a593f (the revision SciPy 1.17.1 pins):
//! `igam`, `igamc`, `igam_fac`, the power series, the continued fraction, `igamc_series` and
//! Temme's `asymptotic_series` of `cephes/igam.h`; `igami` and `igamci` of `cephes/igami.h`
//! (DiDonato & Morris' initial estimate `find_inverse_gamma`, then at most three Halley
//! steps); `lgam1p` and `log1pmx` of `cephes/unity.h`; the coefficient table of
//! `cephes/igam_asymp_coeff.h`. BSD-3-Clause, Copyright SciPy developers; after Cephes
//! `igam.c` (Copyright 1985, 1987 Stephen L. Moshier) and Boost's `igamma_inverse.hpp`
//! (Copyright John Maddock 2006, Boost Software License 1.0). frankenscipy-6fpkm ported the
//! Temme zones, frankenscipy-449uv the rest.
//!
//! SciPy's `gammainc`, `gammaincc`, `gammaincinv` and `gammainccinv` are exactly `igam`,
//! `igamc`, `igami` and `igamci`; `chdtr`/`chdtrc`/`chdtri`, `pdtr`/`pdtrc`/`pdtri` and
//! `gdtr`/`gdtrc` call them with scaled arguments, and its `gdtria`/`gdtrix` divide `igami`.
//!
//! Every operation is in the C's order, every libm call is the C's (`exp`, `log`, `pow`,
//! `log1p` and `sqrt` are glibc's on both sides; fsci's `f64` methods call the same functions)
//! and nothing is fused, so each result is SciPy's to the bit. A Python emulation of exactly
//! this code, in IEEE doubles with glibc reached through ctypes, agreed with SciPy 1.17.1 on
//! all 350,069 evaluations of a grid over every branch and edge of the fourteen functions
//! above (`a` from 1e-3 to 1e5, `x` from 1e-10 to 1e5, inverse targets from 1e-300 to
//! 1 − 1e-17). Three deliberately wrong variants of it (`igam_fac` as a plain `exp` of logs,
//! one ulp on the Lanczos prefactor, one table coefficient moved by 2^-40) disagreed on 15,209,
//! 9,105 and 28 of the 105,756 evaluations of the four core functions, so the comparison can
//! see a difference.
//!
//! The helpers are fsci's existing xsf-exact ones: [`crate::gamma::xsf_lgam`] (Cephes'
//! `lgam`), [`crate::gamma::xsf_gamma`] (Cephes' `Gamma`), Cephes' Lanczos sum
//! [`crate::gamma::cephes_lanczos_sum_expg_scaled`], Cephes' `erfc`
//! ([`crate::error::erfc_scalar`]) and `expm1` ([`crate::convenience::cephes_expm1`]).
//! `lgam1p`'s Taylor series takes `zeta(n, 1)` from [`ZETA_INT`], the values Cephes' `zeta`
//! returns (pinned by a test against [`crate::convenience::hurwitz_zeta`], which is that
//! `zeta`), where xsf recomputes each one with up to 22 `pow` calls per term.
//!
//! ## Temme's expansion
//!
//! SciPy answers two zones with Temme's uniform asymptotic expansion (DLMF 8.12.3/8.12.4)
//! instead of the power series or the continued fraction:
//!
//! ```text
//! 20 < a < 200   and |x − a|/a < 0.3
//! a > 200        and |x − a|/a < 4.5/√a
//! ```
//!
//! There the series needs about `8·√a` terms and the continued fraction about `9·a^(1/3)`,
//! while the expansion needs at most 25 table rows. Each output is summed on its own, as
//! SciPy's `igam` and `igamc` each call `asymptotic_series`, so `Q` is not formed as `1 − P`.
//!
//! The port is bit-for-bit the C: the same operation order, the same `MACHEP` stops, the
//! same `etapow` cache. [`log1pmx`] is Cephes' own series from `xsf/cephes/unity.h`, because
//! fsci's [`crate::convenience::log1pmx_scalar`] sums the same series by a different
//! recurrence and stops at `ε` rather than `ε/2`: it differs from SciPy's by up to 2 ulp at
//! `σ = (x − a)/a` in the zones, and `η = √(−2·log1pmx(σ))` carries that through `erfc` into
//! up to 24 ulp of `P` or `Q` (measured on 4,037 zone points against SciPy 1.17.1). `erfc`
//! is [`crate::error::erfc_scalar`], which is Cephes' `erfc` with the same coefficients.
//! `2π` is `TAU`, which is `2·π` exactly, as `2 * M_PI` is in the C.

use std::f64::consts::{E, TAU};

/// Cephes' `MACHEP`, `2^-53`.
const MACHEP: f64 = 1.11022302462515654042e-16;

/// Cephes' `MAXLOG`, `log(DBL_MAX)`: `igam_fac` answers 0 below `-MAXLOG`.
const MAXLOG: f64 = 7.09782712893383996732e2;

/// Cephes' `MAXITER` for the `log1pmx` series. `|σ| < 0.32` in the zones, where the series
/// stops within 40 terms, so the cap never binds from here.
const LOG1PMX_MAXITER: usize = 500;

/// `igam_MAXITER`, the cap of the power series, the continued fraction and `igamc_series`.
const IGAM_MAXITER: usize = 2000;

/// `igam_big` and `igam_biginv`: the continued fraction rescales its convergents past `2^52`.
const IGAM_BIG: f64 = 4.503599627370496e15;
const IGAM_BIGINV: f64 = 2.22044604925031308085e-16;

/// Cephes' `SCIPY_EULER`, the Euler-Mascheroni constant.
const SCIPY_EULER: f64 = 0.577215664901532860606512090082402431;

/// `igam_SMALL`, `igam_LARGE`, `igam_SMALLRATIO` and `igam_LARGERATIO` of `igam.h`.
const SMALL: f64 = 20.0;
const LARGE: f64 = 200.0;
const SMALL_RATIO: f64 = 0.3;
const LARGE_RATIO: f64 = 4.5;

/// `igam_asymp_coeff_K` and `igam_asymp_coeff_N`.
const COEFF_K: usize = 25;
const COEFF_N: usize = 25;

/// `igam_asymp_coeff_d`: row `k` holds the Taylor coefficients in `η` of Temme's `c_k(η)`
/// (DLMF 8.12.8). Verbatim from `igam_asymp_coeff.h`, in its four-per-line layout so it can
/// be checked against the header line by line.
#[rustfmt::skip]
const COEFF_D: [[f64; COEFF_N]; COEFF_K] = [
    [-3.3333333333333333e-1,  8.3333333333333333e-2,   -1.4814814814814815e-2,  1.1574074074074074e-3,
     3.527336860670194e-4,    -1.7875514403292181e-4,  3.9192631785224378e-5,   -2.1854485106799922e-6,
     -1.85406221071516e-6,    8.296711340953086e-7,    -1.7665952736826079e-7,  6.7078535434014986e-9,
     1.0261809784240308e-8,   -4.3820360184533532e-9,  9.1476995822367902e-10,  -2.551419399494625e-11,
     -5.8307721325504251e-11, 2.4361948020667416e-11,  -5.0276692801141756e-12, 1.1004392031956135e-13,
     3.3717632624009854e-13,  -1.3923887224181621e-13, 2.8534893807047443e-14,  -5.1391118342425726e-16,
     -1.9752288294349443e-15],
    [-1.8518518518518519e-3,  -3.4722222222222222e-3,  2.6455026455026455e-3,   -9.9022633744855967e-4,
     2.0576131687242798e-4,   -4.0187757201646091e-7,  -1.8098550334489978e-5,  7.6491609160811101e-6,
     -1.6120900894563446e-6,  4.6471278028074343e-9,   1.378633446915721e-7,    -5.752545603517705e-8,
     1.1951628599778147e-8,   -1.7543241719747648e-11, -1.0091543710600413e-9,  4.1627929918425826e-10,
     -8.5639070264929806e-11, 6.0672151016047586e-14,  7.1624989648114854e-12,  -2.9331866437714371e-12,
     5.9966963656836887e-13,  -2.1671786527323314e-16, -4.9783399723692616e-14, 2.0291628823713425e-14,
     -4.13125571381061e-15],
    [4.1335978835978836e-3,   -2.6813271604938272e-3,  7.7160493827160494e-4,  2.0093878600823045e-6,
     -1.0736653226365161e-4,  5.2923448829120125e-5,   -1.2760635188618728e-5, 3.4235787340961381e-8,
     1.3721957309062933e-6,   -6.298992138380055e-7,   1.4280614206064242e-7,  -2.0477098421990866e-10,
     -1.4092529910867521e-8,  6.228974084922022e-9,    -1.3670488396617113e-9, 9.4283561590146782e-13,
     1.2872252400089318e-10,  -5.5645956134363321e-11, 1.1975935546366981e-11, -4.1689782251838635e-15,
     -1.0940640427884594e-12, 4.6622399463901357e-13,  -9.905105763906906e-14, 1.8931876768373515e-17,
     8.8592218725911273e-15],
    [6.4943415637860082e-4,   2.2947209362139918e-4,   -4.6918949439525571e-4,  2.6772063206283885e-4,
     -7.5618016718839764e-5,  -2.3965051138672967e-7,  1.1082654115347302e-5,   -5.6749528269915966e-6,
     1.4230900732435884e-6,   -2.7861080291528142e-11, -1.6958404091930277e-7,  8.0994649053880824e-8,
     -1.9111168485973654e-8,  2.3928620439808118e-12,  2.0620131815488798e-9,   -9.4604966618551322e-10,
     2.1541049775774908e-10,  -1.388823336813903e-14,  -2.1894761681963939e-11, 9.7909989511716851e-12,
     -2.1782191880180962e-12, 6.2088195734079014e-17,  2.126978363279737e-13,   -9.3446887915174333e-14,
     2.0453671226782849e-14],
    [-8.618882909167117e-4,   7.8403922172006663e-4,   -2.9907248030319018e-4, -1.4638452578843418e-6,
     6.6414982154651222e-5,   -3.9683650471794347e-5,  1.1375726970678419e-5,  2.5074972262375328e-10,
     -1.6954149536558306e-6,  8.9075075322053097e-7,   -2.2929348340008049e-7, 2.956794137544049e-11,
     2.8865829742708784e-8,   -1.4189739437803219e-8,  3.4463580499464897e-9,  -2.3024517174528067e-13,
     -3.9409233028046405e-10, 1.8602338968504502e-10,  -4.356323005056618e-11, 1.2786001016296231e-15,
     4.6792750266579195e-12,  -2.1492464706134829e-12, 4.9088156148096522e-13, -6.3385914848915603e-18,
     -5.0453320690800944e-14],
    [-3.3679855336635815e-4,  -6.9728137583658578e-5,  2.7727532449593921e-4,  -1.9932570516188848e-4,
     6.7977804779372078e-5,   1.419062920643967e-7,    -1.3594048189768693e-5, 8.0184702563342015e-6,
     -2.2914811765080952e-6,  -3.252473551298454e-10,  3.4652846491085265e-7,  -1.8447187191171343e-7,
     4.8240967037894181e-8,   -1.7989466721743515e-14, -6.3061945000135234e-9, 3.1624176287745679e-9,
     -7.8409242536974293e-10, 5.1926791652540407e-15,  9.3589442423067836e-11, -4.5134262161632782e-11,
     1.0799129993116827e-11,  -3.661886712685252e-17,  -1.210902069055155e-12, 5.6807435849905643e-13,
     -1.3249659916340829e-13],
    [5.3130793646399222e-4,   -5.9216643735369388e-4,  2.7087820967180448e-4,   7.9023532326603279e-7,
     -8.1539693675619688e-5,  5.6116827531062497e-5,   -1.8329116582843376e-5,  -3.0796134506033048e-9,
     3.4651553688036091e-6,   -2.0291327396058604e-6,  5.7887928631490037e-7,   2.338630673826657e-13,
     -8.8286007463304835e-8,  4.7435958880408128e-8,   -1.2545415020710382e-8,  8.6496488580102925e-14,
     1.6846058979264063e-9,   -8.5754928235775947e-10, 2.1598224929232125e-10,  -7.6132305204761539e-16,
     -2.6639822008536144e-11, 1.3065700536611057e-11,  -3.1799163902367977e-12, 4.7109761213674315e-18,
     3.6902800842763467e-13],
    [3.4436760689237767e-4,   5.1717909082605922e-5,   -3.3493161081142236e-4,  2.812695154763237e-4,
     -1.0976582244684731e-4,  -1.2741009095484485e-7,  2.7744451511563644e-5,   -1.8263488805711333e-5,
     5.7876949497350524e-6,   4.9387589339362704e-10,  -1.0595367014026043e-6,  6.1667143761104075e-7,
     -1.7562973359060462e-7,  -1.2974473287015439e-12, 2.695423606288966e-8,    -1.4578352908731271e-8,
     3.887645959386175e-9,    -3.8810022510194121e-17, -5.3279941738772867e-10, 2.7437977643314845e-10,
     -6.9957960920705679e-11, 2.5899863874868481e-17,  8.8566890996696381e-12,  -4.403168815871311e-12,
     1.0865561947091654e-12],
    [-6.5262391859530942e-4, 8.3949872067208728e-4,   -4.3829709854172101e-4, -6.969091458420552e-7,
     1.6644846642067548e-4,  -1.2783517679769219e-4,  4.6299532636913043e-5,  4.5579098679227077e-9,
     -1.0595271125805195e-5, 6.7833429048651666e-6,   -2.1075476666258804e-6, -1.7213731432817145e-11,
     3.7735877416110979e-7,  -2.1867506700122867e-7,  6.2202288040189269e-8,  6.5977038267330006e-16,
     -9.5903864974256858e-9, 5.2132144922808078e-9,   -1.3991589583935709e-9, 5.382058999060575e-16,
     1.9484714275467745e-10, -1.0127287556389682e-10, 2.6077347197254926e-11, -5.0904186999932993e-18,
     -3.3721464474854592e-12],
    [-5.9676129019274625e-4, -7.2048954160200106e-5,  6.7823088376673284e-4,   -6.4014752602627585e-4,
     2.7750107634328704e-4,  1.8197008380465151e-7,   -8.4795071170685032e-5,  6.105192082501531e-5,
     -2.1073920183404862e-5, -8.8585890141255994e-10, 4.5284535953805377e-6,   -2.8427815022504408e-6,
     8.7082341778646412e-7,  3.6886101871706965e-12,  -1.5344695190702061e-7,  8.862466778790695e-8,
     -2.5184812301826817e-8, -1.0225912098215092e-14, 3.8969470758154777e-9,   -2.1267304792235635e-9,
     5.7370135528051385e-10, -1.887749850169741e-19,  -8.0931538694657866e-11, 4.2382723283449199e-11,
     -1.1002224534207726e-11],
    [1.3324454494800656e-3,  -1.9144384985654775e-3, 1.1089369134596637e-3,   9.932404122642299e-7,
     -5.0874501293093199e-4, 4.2735056665392884e-4,  -1.6858853767910799e-4,  -8.1301893922784998e-9,
     4.5284402370562147e-5,  -3.127053674781734e-5,  1.044986828530338e-5,    4.8435226265680926e-11,
     -2.1482565873456258e-6, 1.329369701097492e-6,   -4.0295693092101029e-7,  -1.7567877666323291e-13,
     7.0145043163668257e-8,  -4.040787734999483e-8,  1.1474026743371963e-8,   3.9642746853563325e-18,
     -1.7804938269892714e-9, 9.7480262548731646e-10, -2.6405338676507616e-10, 5.794875163403742e-18,
     3.7647749553543836e-11],
    [1.579727660730835e-3,   1.6251626278391582e-4,   -2.0633421035543276e-3, 2.1389686185689098e-3,
     -1.0108559391263003e-3, -3.9912705529919201e-7,  3.6235025084764691e-4,  -2.8143901463712154e-4,
     1.0449513336495887e-4,  2.1211418491830297e-9,   -2.5779417251947842e-5, 1.7281818956040463e-5,
     -5.6413773872904282e-6, -1.1024320105776174e-11, 1.1223224418895175e-6,  -6.8693396379526735e-7,
     2.0653236975414887e-7,  4.6714772409838506e-14,  -3.5609886164949055e-8, 2.0470855345905963e-8,
     -5.8091738633283358e-9, -1.332821287582869e-16,  9.0354604391335133e-10, -4.9598782517330834e-10,
     1.3481607129399749e-10],
    [-4.0725121195140166e-3, 6.4033628338080698e-3,  -4.0410161081676618e-3, -2.183732802866233e-6,
     2.1740441801254639e-3,  -1.9700440518418892e-3, 8.3595469747962458e-4,  1.9445447567109655e-8,
     -2.5779387120421696e-4, 1.9009987368139304e-4,  -6.7696499937438965e-5, -1.4440629666426572e-10,
     1.5712512518742269e-5,  -1.0304008744776893e-5, 3.304517767401387e-6,   7.9829760242325709e-13,
     -6.4097794149313004e-7, 3.8894624761300056e-7,  -1.1618347644948869e-7, -2.816808630596451e-15,
     1.9878012911297093e-8,  -1.1407719956357511e-8, 3.2355857064185555e-9,  4.1759468293455945e-20,
     -5.0423112718105824e-10],
    [-5.9475779383993003e-3, -5.4016476789260452e-4,  8.7910413550767898e-3,  -9.8576315587856125e-3,
     5.0134695031021538e-3,  1.2807521786221875e-6,   -2.0626019342754683e-3, 1.7109128573523058e-3,
     -6.7695312714133799e-4, -6.9011545676562133e-9,  1.8855128143995902e-4,  -1.3395215663491969e-4,
     4.6263183033528039e-5,  4.0034230613321351e-11,  -1.0255652921494033e-5, 6.612086372797651e-6,
     -2.0913022027253008e-6, -2.0951775649603837e-13, 3.9756029041993247e-7,  -2.3956211978815887e-7,
     7.1182883382145864e-8,  8.925574873053455e-16,   -1.2101547235064676e-8, 6.9350618248334386e-9,
     -1.9661464453856102e-9],
    [1.7402027787522711e-2,  -2.9527880945699121e-2, 2.0045875571402799e-2,  7.0289515966903407e-6,
     -1.2375421071343148e-2, 1.1976293444235254e-2,  -5.4156038466518525e-3, -6.3290893396418616e-8,
     1.8855118129005065e-3,  -1.473473274825001e-3,  5.5515810097708387e-4,  5.2406834412550662e-10,
     -1.4357913535784836e-4, 9.9181293224943297e-5,  -3.3460834749478311e-5, -3.5755837291098993e-12,
     7.1560851960630076e-6,  -4.5516802628155526e-6, 1.4236576649271475e-6,  1.8803149082089664e-14,
     -2.6623403898929211e-7, 1.5950642189595716e-7,  -4.7187514673841102e-8, -6.5107872958755177e-17,
     7.9795091026746235e-9],
    [3.0249124160905891e-2,  2.4817436002649977e-3,  -4.9939134373457022e-2, 5.9915643009307869e-2,
     -3.2483207601623391e-2, -5.7212968652103441e-6, 1.5085251778569354e-2,  -1.3261324005088445e-2,
     5.5515262632426148e-3,  3.0263182257030016e-8,  -1.7229548406756723e-3, 1.2893570099929637e-3,
     -4.6845138348319876e-4, -1.830259937893045e-10, 1.1449739014822654e-4,  -7.7378565221244477e-5,
     2.5625836246985201e-5,  1.0766165333192814e-12, -5.3246809282422621e-6, 3.349634863064464e-6,
     -1.0381253128684018e-6, -5.608909920621128e-15, 1.9150821930676591e-7,  -1.1418365800203486e-7,
     3.3654425209171788e-8],
    [-9.9051020880159045e-2, 1.7954011706123486e-1,  -1.2989606383463778e-1, -3.1478872752284357e-5,
     9.0510635276848131e-2,  -9.2828824411184397e-2, 4.4412112839877808e-2,  2.7779236316835888e-7,
     -1.7229543805449697e-2, 1.4182925050891573e-2,  -5.6214161633747336e-3, -2.39598509186381e-9,
     1.6029634366079908e-3,  -1.1606784674435773e-3, 4.1001337768153873e-4,  1.8365800754090661e-11,
     -9.5844256563655903e-5, 6.3643062337764708e-5,  -2.076250624489065e-5,  -1.1806020912804483e-13,
     4.2131808239120649e-6,  -2.6262241337012467e-6, 8.0770620494930662e-7,  6.0125912123632725e-16,
     -1.4729737374018841e-7],
    [-1.9994542198219728e-1, -1.5056113040026424e-2,  3.6470239469348489e-1,  -4.6435192311733545e-1,
     2.6640934719197893e-1,  3.4038266027147191e-5,   -1.3784338709329624e-1, 1.276467178337056e-1,
     -5.6213828755200985e-2, -1.753150885483011e-7,   1.9235592956768113e-2,  -1.5088821281095315e-2,
     5.7401854451350123e-3,  1.0622382710310225e-9,   -1.5335082692563998e-3, 1.0819320643228214e-3,
     -3.7372510193945659e-4, -6.6170909729031985e-12, 8.4263617380909628e-5,  -5.5150706827483479e-5,
     1.7769536448348069e-5,  3.8827923210205533e-14,  -3.53513697488768e-6,   2.1865832130045269e-6,
     -6.6812849447625594e-7],
    [7.2438608504029431e-1,  -1.3918010932653375,    1.0654143352413968,     1.876173868950258e-4,
     -8.2705501176152696e-1, 8.9352433347828414e-1,  -4.4971003995291339e-1, -1.6107401567546652e-6,
     1.9235590165271091e-1,  -1.6597702160042609e-1, 6.8882222681814333e-2,  1.3910091724608687e-8,
     -2.146911561508663e-2,  1.6228980898865892e-2,  -5.9796016172584256e-3, -1.1287469112826745e-10,
     1.5167451119784857e-3,  -1.0478634293553899e-3, 3.5539072889126421e-4,  8.1704322111801517e-13,
     -7.7773013442452395e-5, 5.0291413897007722e-5,  -1.6035083867000518e-5, 1.2469354315487605e-14,
     3.1369106244517615e-6],
    [1.6668949727276811,     1.165462765994632e-1,   -3.3288393225018906,    4.4692325482864037,
     -2.6977693045875807,    -2.600667859891061e-4,  1.5389017615694539,     -1.4937962361134612,
     6.8881964633233148e-1,  1.3077482004552385e-6,  -2.5762963325596288e-1, 2.1097676102125449e-1,
     -8.3714408359219882e-2, -7.7920428881354753e-9, 2.4267923064833599e-2,  -1.7813678334552311e-2,
     6.3970330388900056e-3,  4.9430807090480523e-11, -1.5554602758465635e-3, 1.0561196919903214e-3,
     -3.5277184460472902e-4, 9.3002334645022459e-14, 7.5285855026557172e-5,  -4.8186515569156351e-5,
     1.5227271505597605e-5],
    [-6.6188298861372935,    1.3397985455142589e+1,  -1.0789350606845146e+1, -1.4352254537875018e-3,
     9.2333694596189809,     -1.0456552819547769e+1, 5.5105526029033471,     1.2024439690716742e-5,
     -2.5762961164755816,    2.3207442745387179,     -1.0045728797216284,    -1.0207833290021914e-7,
     3.3975092171169466e-1,  -2.6720517450757468e-1, 1.0235252851562706e-1,  8.4329730484871625e-10,
     -2.7998284958442595e-2, 2.0066274144976813e-2,  -7.0554368915086242e-3, 1.9402238183698188e-12,
     1.6562888105449611e-3,  -1.1082898580743683e-3, 3.654545161310169e-4,   -5.1290032026971794e-11,
     -7.6340103696869031e-5],
    [-1.7112706061976095e+1, -1.1208044642899116,     3.7131966511885444e+1,  -5.2298271025348962e+1,
     3.3058589696624618e+1,  2.4791298976200222e-3,   -2.061089403411526e+1,  2.088672775145582e+1,
     -1.0045703956517752e+1, -1.2238783449063012e-5,  4.0770134274221141,     -3.473667358470195,
     1.4329352617312006,     7.1359914411879712e-8,   -4.4797257159115612e-1, 3.4112666080644461e-1,
     -1.2699786326594923e-1, -2.8953677269081528e-10, 3.3125776278259863e-2,  -2.3274087021036101e-2,
     8.0399993503648882e-3,  -1.177805216235265e-9,   -1.8321624891071668e-3, 1.2108282933588665e-3,
     -3.9479941246822517e-4],
    [7.389033153567425e+1,   -1.5680141270402273e+2, 1.322177542759164e+2,   1.3692876877324546e-2,
     -1.2366496885920151e+2, 1.4620689391062729e+2,  -8.0365587724865346e+1, -1.1259851148881298e-4,
     4.0770132196179938e+1,  -3.8210340013273034e+1, 1.719522294277362e+1,   9.3519707955168356e-7,
     -6.2716159907747034,    5.1168999071852637,     -2.0319658112299095,    -4.9507215582761543e-9,
     5.9626397294332597e-1,  -4.4220765337238094e-1, 1.6079998700166273e-1,  -2.4733786203223402e-8,
     -4.0307574759979762e-2, 2.7849050747097869e-2,  -9.4751858992054221e-3, 6.419922235909132e-6,
     2.1250180774699461e-3],
    [2.1216837098382522e+2,  1.3107863022633868e+1,  -4.9698285932871748e+2, 7.3121595266969204e+2,
     -4.8213821720890847e+2, -2.8817248692894889e-2, 3.2616720302947102e+2,  -3.4389340280087117e+2,
     1.7195193870816232e+2,  1.4038077378096158e-4,  -7.52594195897599e+1,   6.651969984520934e+1,
     -2.8447519748152462e+1, -7.613702615875391e-7,  9.5402237105304373,     -7.5175301113311376,
     2.8943997568871961,     -4.6612194999538201e-7, -8.0615149598794088e-1, 5.8483006570631029e-1,
     -2.0845408972964956e-1, 1.4765818959305817e-4,  5.1000433863753019e-2,  -3.3066252141883665e-2,
     1.5109265210467774e-2],
    [-9.8959643098322368e+2, 2.1925555360905233e+3,  -1.9283586782723356e+3, -1.5925738122215253e-1,
     1.9569985945919857e+3,  -2.4072514765081556e+3, 1.3756149959336496e+3,  1.2920735237496668e-3,
     -7.525941715948055e+2,  7.3171668742208716e+2,  -3.4137023466220065e+2, -9.9857390260608043e-6,
     1.3356313181291573e+2,  -1.1276295161252794e+2, 4.6310396098204458e+1,  -7.9237387133614756e-6,
     -1.4510726927018646e+1, 1.1111771248100563e+1,  -4.1690817945270892,    3.1008219800117808e-3,
     1.1220095449981468,     -7.6052379926149916e-1, 3.6262236505085254e-1,  2.216867741940747e-1,
     4.8683443692930507e-1],
];

/// `zeta(n, 1)` for `n = 2..=41` as xsf's Cephes `zeta` returns it (not always the nearest
/// double: `zeta(2, 1)` is one ulp above `π²/6`). `lgam1p`'s Taylor series takes its
/// coefficients from here where the C calls `zeta(n, 1)` for each term; the test
/// `zeta_int_is_cephes_zeta_bit_for_bit` pins every entry to [`crate::convenience::hurwitz_zeta`].
const ZETA_INT: [f64; 40] = [
    1.6449340668482266,
    1.202056903159594,
    1.0823232337111381,
    1.0369277551433704,
    1.0173430619844488,
    1.008349277381923,
    1.0040773561979446,
    1.0020083928260826,
    1.0009945751278182,
    1.0004941886041194,
    1.0002460865533078,
    1.0001227133475785,
    1.0000612481350586,
    1.0000305882363072,
    1.0000152822594084,
    1.0000076371976376,
    1.000003817293265,
    1.0000019082127163,
    1.0000009539620338,
    1.0000004769329867,
    1.0000002384505027,
    1.000000119219926,
    1.000000059608189,
    1.0000000298035034,
    1.0000000149015549,
    1.0000000074507118,
    1.000000003725334,
    1.0000000018626598,
    1.0000000009313275,
    1.0000000004656628,
    1.000000000232831,
    1.0000000001164155,
    1.0000000000582077,
    1.0000000000291038,
    1.000000000014552,
    1.000000000007276,
    1.000000000003638,
    1.000000000001819,
    1.0000000000009095,
    1.0000000000004547,
];

#[cfg(test)]
thread_local! {
    /// Calls that took the Temme branch on this thread: the must-hit control of
    /// frankenscipy-6fpkm. Thread-local so concurrently running tests cannot see each other.
    pub(crate) static TEMME_HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The `PATH_*` branches taken on this thread since the last [`take_paths`]: the must-hit
    /// control of the branch tests (frankenscipy-449uv).
    static PATHS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Branches of `igam.h` and `igami.h`, recorded by [`took`] in test builds only.
const PATH_SERIES: u32 = 1;
const PATH_CF: u32 = 1 << 1;
const PATH_IGAMC_SERIES: u32 = 1 << 2;
const PATH_TEMME: u32 = 1 << 3;
const PATH_FAC_UNDERFLOW: u32 = 1 << 4;
const PATH_FAC_LOGSPACE: u32 = 1 << 5;
const PATH_FAC_LANCZOS: u32 = 1 << 6;
const PATH_FAC_LOG1PMX: u32 = 1 << 7;
const PATH_A1: u32 = 1 << 8;
const PATH_EQ21_POW: u32 = 1 << 9;
const PATH_EQ21_EXP: u32 = 1 << 10;
const PATH_EQ22: u32 = 1 << 11;
const PATH_EQ23: u32 = 1 << 12;
const PATH_EQ24: u32 = 1 << 13;
const PATH_EQ25_SMALL_A: u32 = 1 << 14;
const PATH_EQ31_LARGE_A: u32 = 1 << 15;
const PATH_EQ31_UPPER: u32 = 1 << 16;
const PATH_EQ25_LARGE_A: u32 = 1 << 17;
const PATH_EQ33: u32 = 1 << 18;
const PATH_EQ35: u32 = 1 << 19;
const PATH_EQ36: u32 = 1 << 20;
const PATH_Z_DIRECT: u32 = 1 << 21;
const PATH_HALLEY_FAC0: u32 = 1 << 22;
const PATH_NEWTON: u32 = 1 << 23;
const PATH_SWITCH: u32 = 1 << 24;

#[cfg(test)]
fn took(path: u32) {
    PATHS.with(|paths| paths.set(paths.get() | path));
}

#[cfg(not(test))]
#[inline(always)]
fn took(_path: u32) {}

/// The branches taken on this thread since the last call, and a reset.
#[cfg(test)]
fn take_paths() -> u32 {
    PATHS.with(|paths| paths.replace(0))
}

/// Cephes' `polevl`: `coef[0]·xᴺ + … + coef[N]` by Horner, in the C's order.
fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef[1..].iter().fold(coef[0], |ans, &c| ans * x + c)
}

/// SciPy's zone test in `igam` and `igamc`: `true` where both answer with
/// `asymptotic_series`. `false` for every NaN, zero, infinite or negative input, since each
/// comparison is then false or the ratio is at least 1.
pub(crate) fn in_temme_zone(a: f64, x: f64) -> bool {
    let absxma_a = (x - a).abs() / a;
    (a > SMALL && a < LARGE && absxma_a < SMALL_RATIO)
        || (a > LARGE && absxma_a < LARGE_RATIO / a.sqrt())
}

/// `P(a, x)` (`lower`) or `Q(a, x)` from Temme's expansion, `asymptotic_series` of `igam.h`
/// (DLMF 8.12.3/8.12.4). The caller has checked [`in_temme_zone`].
///
/// SciPy's `asymptotic_series(a, x, IGAM)` and `asymptotic_series(a, x, IGAMC)` differ only
/// in `sgn = ∓1`, which enters as `erfc(sgn·η·√(a/2))` and `+ sgn·e^(−aη²/2)·Σ/√(2πa)`.
/// Negation is exact in both places, so each output below is bit-for-bit the C's.
fn asymptotic_series(a: f64, x: f64, lower: bool) -> f64 {
    took(PATH_TEMME);
    #[cfg(test)]
    TEMME_HITS.with(|hits| hits.set(hits.get() + 1));

    let lambda = x / a;
    let sigma = (x - a) / a;
    let eta = if lambda > 1.0 {
        (-2.0 * log1pmx(sigma)).sqrt()
    } else if lambda < 1.0 {
        -(-2.0 * log1pmx(sigma)).sqrt()
    } else {
        0.0
    };
    // The C forms sgn·η·√(a/2) as (sgn·η)·√(a/2), which is ±(η·√(a/2)) exactly.
    let z = eta * (a / 2.0).sqrt();

    let mut etapow = [0.0_f64; COEFF_N];
    etapow[0] = 1.0;
    let mut maxpow = 0_usize;
    let mut sum = 0.0_f64;
    let mut afac = 1.0_f64;
    let mut absoldterm = f64::INFINITY;
    for row in &COEFF_D {
        let mut ck = row[0];
        for n in 1..COEFF_N {
            if n > maxpow {
                etapow[n] = eta * etapow[n - 1];
                maxpow += 1;
            }
            let ckterm = row[n] * etapow[n];
            ck += ckterm;
            if ckterm.abs() < MACHEP * ck.abs() {
                break;
            }
        }
        let term = ck * afac;
        let absterm = term.abs();
        if absterm > absoldterm {
            break;
        }
        sum += term;
        if absterm < MACHEP * sum.abs() {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    // The C's (sgn·e)·Σ/√(2πa) is ±(e·Σ/√(2πa)) exactly, and r + (−t) is r − t.
    let t = (-0.5 * a * eta * eta).exp() * sum / (TAU * a).sqrt();
    if lower {
        0.5 * crate::error::erfc_scalar(-z) - t
    } else {
        0.5 * crate::error::erfc_scalar(z) + t
    }
}

/// `ln(1 + x) − x` as `xsf/cephes/unity.h` computes it: the series
/// `Σ_{n≥2} (−1)^(n+1)·xⁿ/n` with the power carried as a running product, stopped when a
/// term drops below `MACHEP` of the sum, for `|x| < 0.5`, and Cephes' `log1p(x) − x`
/// otherwise. Cephes' `log1p` is `log(1 + x)` whenever `1 + x` is outside `[√½, √2]`, which
/// `|x| ≥ 0.5` always is. Temme's zones keep `|σ| ≤ 4.5/√200 < 0.32` and `igam_fac` keeps its
/// argument below 0.41 in magnitude, so only the series runs from this module.
fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res = 0.0_f64;
        for n in 2..LOG1PMX_MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
        }
        res
    } else {
        (1.0 + x).ln() - x
    }
}

/// `xᵃ·e⁻ˣ / Γ(a)`, `igam_fac` of `igam.h`: the factor the series and the continued fraction
/// scale by, and the derivative the inverses step with. Where `|a − x| > 0.4·|a|` it is the
/// `exp` of a sum of logs, and 0 once that sum is below `−MAXLOG`; nearer the diagonal, where
/// those logs cancel, it is Boost's Lanczos form (Maddock et al., equations 15 and 16 with
/// `exp(x − a)` corrected to `exp(a − x)`), through `pow` below 200 and `log1pmx` above.
fn igam_fac(a: f64, x: f64) -> f64 {
    if (a - x).abs() > 0.4 * a.abs() {
        let ax = a * x.ln() - x - crate::gamma::xsf_lgam(a);
        if ax < -MAXLOG {
            took(PATH_FAC_UNDERFLOW);
            return 0.0;
        }
        took(PATH_FAC_LOGSPACE);
        return ax.exp();
    }
    let g = crate::gamma::CEPHES_LANCZOS_G;
    let fac = a + g - 0.5;
    // The C's `std::exp(1)` is e correctly rounded, which `E` is.
    let mut res = (fac / E).sqrt() / crate::gamma::cephes_lanczos_sum_expg_scaled(a);
    if a < 200.0 && x < 200.0 {
        took(PATH_FAC_LANCZOS);
        res *= (a - x).exp() * (x / fac).powf(a);
    } else {
        took(PATH_FAC_LOG1PMX);
        let num = x - a - g + 0.5;
        res *= (a * log1pmx(num / fac) + x * (0.5 - g) / fac).exp();
    }
    res
}

/// `Q(a, x)` by the continued fraction of DLMF 8.9.2, `igamc_continued_fraction` of
/// `igam.h`: convergents `pk/qk`, rescaled by `2^-52` whenever `|pk|` passes `2^52`, until
/// two successive ones agree to `MACHEP`.
fn igamc_continued_fraction(a: f64, x: f64) -> f64 {
    took(PATH_CF);
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0_f64;
    let mut pkm2 = 1.0_f64;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    for _ in 0..IGAM_MAXITER {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > IGAM_BIG {
            pkm2 *= IGAM_BIGINV;
            pkm1 *= IGAM_BIGINV;
            qkm2 *= IGAM_BIGINV;
            qkm1 *= IGAM_BIGINV;
        }
        if t <= MACHEP {
            break;
        }
    }
    ans * ax
}

/// `P(a, x)` by the power series of DLMF 8.11.4, `igam_series` of `igam.h`.
fn igam_series(a: f64, x: f64) -> f64 {
    took(PATH_SERIES);
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut r = a;
    let mut c = 1.0_f64;
    let mut ans = 1.0_f64;
    for _ in 0..IGAM_MAXITER {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c <= MACHEP * ans {
            break;
        }
    }
    ans * ax / a
}

/// `Q(a, x)` for small `x` by DLMF 8.7.3, `igamc_series` of `igam.h`: the series of
/// `igam_series` rearranged so that `1 − xᵃ/Γ(a+1)` is formed by `expm1` and does not cancel.
fn igamc_series(a: f64, x: f64) -> f64 {
    took(PATH_IGAMC_SERIES);
    let mut fac = 1.0_f64;
    let mut sum = 0.0_f64;
    for n in 1..IGAM_MAXITER {
        let n = n as f64;
        fac *= -x / n;
        let term = fac / (a + n);
        sum += term;
        if term.abs() <= MACHEP * sum.abs() {
            break;
        }
    }
    let logx = x.ln();
    let term = -crate::convenience::cephes_expm1(a * logx - lgam1p(a));
    term - (a * logx - crate::gamma::xsf_lgam(a)).exp() * sum
}

/// `ln Γ(1 + x)`, `lgam1p` of `unity.h`: its Taylor series about 0 for `|x| ≤ 0.5`, about 1
/// (plus `ln x`) for `|x − 1| < 0.5`, and Cephes' `lgam(x + 1)` otherwise.
fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        crate::gamma::xsf_lgam(x + 1.0)
    }
}

/// `lgam1p_taylor` of `unity.h`: `−γx + Σ_{n=2}^{41} ζ(n)·(−x)ⁿ/n`, stopped once a term is
/// below `MACHEP` of the sum.
fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -SCIPY_EULER * x;
    let mut xfac = -x;
    for (k, &zeta) in ZETA_INT.iter().enumerate() {
        xfac *= -x;
        let coeff = zeta * xfac / (k + 2) as f64;
        res += coeff;
        if coeff.abs() < MACHEP * res.abs() {
            break;
        }
    }
    res
}

/// The regularized lower incomplete gamma function `P(a, x)`, `igam` of `igam.h`: SciPy's
/// `gammainc`.
///
/// NaN in, NaN out; a negative `a` or `x` is NaN (SciPy's domain error). `a = 0` is 1 for
/// `x > 0` and NaN at `x = 0`; `x = 0` is 0; an infinite `a` is 0, or NaN with `x` infinite
/// too; an infinite `x` is 1. Then Temme's zones, `1 − Q` for `x > max(1, a)`, and the power
/// series everywhere else.
pub(crate) fn igam(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        return if x > 0.0 { 1.0 } else { f64::NAN };
    }
    if x == 0.0 {
        return 0.0;
    }
    if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 0.0 };
    }
    if x.is_infinite() {
        return 1.0;
    }
    if in_temme_zone(a, x) {
        return asymptotic_series(a, x, true);
    }
    if x > 1.0 && x > a {
        return 1.0 - igamc(a, x);
    }
    igam_series(a, x)
}

/// The regularized upper incomplete gamma function `Q(a, x)`, `igamc` of `igam.h`: SciPy's
/// `gammaincc`.
///
/// The edges are `igam`'s complemented (`a = 0` is 0 for `x > 0`, `x = 0` is 1, an infinite
/// `a` is 1, an infinite `x` is 0; NaN where `igam` is NaN). Then Temme's zones; for
/// `x > 1.1` the continued fraction, or `1 − P` when `x < a`; for smaller `x`, `igamc_series`
/// unless `a` is large enough (`a > −0.4/ln x` up to 0.5, `a > 1.1·x` above) for `1 − P`.
pub(crate) fn igamc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    }
    if x == 0.0 {
        return 1.0;
    }
    if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 1.0 };
    }
    if x.is_infinite() {
        return 0.0;
    }
    if in_temme_zone(a, x) {
        return asymptotic_series(a, x, false);
    }
    if x > 1.1 {
        if x < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_continued_fraction(a, x)
        }
    } else if x <= 0.5 {
        if -0.4 / x.ln() < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_series(a, x)
        }
    } else if x * 1.1 < a {
        1.0 - igam_series(a, x)
    } else {
        igamc_series(a, x)
    }
}

/// DiDonato & Morris' equation 32, `find_inverse_s` of `igami.h`: the normal deviate `s` with
/// `Φ(s) ≈ q`, from a rational in `t = √(−2 ln min(p, q))`.
fn find_inverse_s(p: f64, q: f64) -> f64 {
    const A: [f64; 4] = [
        0.213623493715853,
        4.28342155967104,
        11.6616720288968,
        3.31125922108741,
    ];
    const B: [f64; 5] = [
        0.3611708101884203e-1,
        1.27364489782223,
        6.40691597760039,
        6.61053765625462,
        1.0,
    ];
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * q.ln()).sqrt()
    };
    let s = t - polevl(t, &A) / polevl(t, &B);
    if p < 0.5 { -s } else { s }
}

/// DiDonato & Morris' equation 34, `didonato_SN` of `igami.h` with its `N = 100` and
/// `tolerance = 1e-4` (its one call site's): `1 + Σ_{i=1}^{N} xⁱ/((a+1)…(a+i))`, stopped once a
/// term is below the tolerance.
fn didonato_sn(a: f64, x: f64) -> f64 {
    const N: u32 = 100;
    const TOLERANCE: f64 = 1e-4;
    let mut sum = 1.0_f64;
    let mut partial = x / (a + 1.0);
    sum += partial;
    for i in 2..=N {
        partial *= x / (a + f64::from(i));
        sum += partial;
        if partial < TOLERANCE {
            break;
        }
    }
    sum
}

/// DiDonato & Morris' equation 25, which `find_inverse_gamma` evaluates twice: the
/// asymptotic root of `Q(a, x) = q` in `y = −ln(q·Γ(a))` (the caller forms `y`).
fn didonato_eq25(a: f64, y: f64) -> f64 {
    let c1 = (a - 1.0) * y.ln();
    let c1_2 = c1 * c1;
    let c1_3 = c1_2 * c1;
    let c1_4 = c1_2 * c1_2;
    let a_2 = a * a;
    let a_3 = a_2 * a;

    let c2 = (a - 1.0) * (1.0 + c1);
    let c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
    let c4 = (a - 1.0)
        * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0
            + (a_2 - 6.0 * a + 7.0) * c1
            + (11.0 * a_2 - 46.0 * a + 47.0) / 6.0);
    let c5 = (a - 1.0)
        * (-(c1_4 / 4.0)
            + (11.0 * a - 17.0) * c1_3 / 6.0
            + (-3.0 * a_2 + 13.0 * a - 13.0) * c1_2
            + (2.0 * a_3 - 25.0 * a_2 + 72.0 * a - 61.0) * c1 / 2.0
            + (25.0 * a_3 - 195.0 * a_2 + 477.0 * a - 379.0) / 12.0);

    let y_2 = y * y;
    let y_3 = y_2 * y;
    let y_4 = y_2 * y_2;
    y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4)
}

/// DiDonato & Morris' initial estimate of the `x` with `P(a, x) = p`, `Q(a, x) = q`
/// (`q = 1 − p` as the caller formed it), `find_inverse_gamma` of `igami.h`: closed forms for
/// `a = 1`, equations 21 to 25 for `a < 1` by the size of `q·Γ(a)`, and for `a > 1` the
/// Cornish-Fisher-like equation 31, refined by 25 or 33 in the upper tail and by 35 and 36 in
/// the lower.
fn find_inverse_gamma(a: f64, p: f64, q: f64) -> f64 {
    if a == 1.0 {
        took(PATH_A1);
        return if q > 0.9 { -(-p).ln_1p() } else { -q.ln() };
    }
    if a < 1.0 {
        let g = crate::gamma::xsf_gamma(a);
        let b = q * g;
        if b > 0.6 || (b >= 0.45 && a >= 0.3) {
            // Equation 21, with Boost's second form where the first is unstable (p near 1).
            let u = if b * q > 1e-8 && q > 1e-5 {
                took(PATH_EQ21_POW);
                (p * g * a).powf(1.0 / a)
            } else {
                took(PATH_EQ21_EXP);
                ((-q / a) - SCIPY_EULER).exp()
            };
            return u / (1.0 - (u / (a + 1.0)));
        }
        if a < 0.3 && b >= 0.35 {
            took(PATH_EQ22);
            let t = (-SCIPY_EULER - b).exp();
            let u = t * t.exp();
            return t * u.exp();
        }
        if b > 0.15 || a >= 0.3 {
            took(PATH_EQ23);
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            return y - (1.0 - a) * u.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln();
        }
        if b > 0.1 {
            took(PATH_EQ24);
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            return y
                - (1.0 - a) * u.ln()
                - ((u * u + 2.0 * (3.0 - a) * u + (2.0 - a) * (3.0 - a))
                    / (u * u + (5.0 - a) * u + 2.0))
                    .ln();
        }
        took(PATH_EQ25_SMALL_A);
        return didonato_eq25(a, -b.ln());
    }

    // Equation 31.
    let s = find_inverse_s(p, q);
    let s_2 = s * s;
    let s_3 = s_2 * s;
    let s_4 = s_2 * s_2;
    let s_5 = s_4 * s;
    let ra = a.sqrt();

    let mut w = a + s * ra + (s_2 - 1.0) / 3.0;
    w += (s_3 - 7.0 * s) / (36.0 * ra);
    w -= (3.0 * s_4 + 7.0 * s_2 - 16.0) / (810.0 * a);
    w += (9.0 * s_5 + 256.0 * s_3 - 433.0 * s) / (38880.0 * a * ra);

    if a >= 500.0 && (1.0 - w / a).abs() < 1e-6 {
        took(PATH_EQ31_LARGE_A);
        return w;
    }
    if p > 0.5 {
        if w < 3.0 * a {
            took(PATH_EQ31_UPPER);
            return w;
        }
        // C `fmax(2, a·(a − 1))`; `a > 1` here, so no NaN reaches it from a valid `a`.
        let d = 2.0_f64.max(a * (a - 1.0));
        let lg = crate::gamma::xsf_lgam(a);
        let lb = q.ln() + lg;
        if lb < -d * 2.3 {
            took(PATH_EQ25_LARGE_A);
            return didonato_eq25(a, -lb);
        }
        took(PATH_EQ33);
        let u = -lb + (a - 1.0) * w.ln() - (1.0 + (1.0 - a) / (1.0 + w)).ln();
        return -lb + (a - 1.0) * u.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln();
    }

    let mut z = w;
    let ap1 = a + 1.0;
    let ap2 = a + 2.0;
    if w < 0.15 * ap1 {
        // Equation 35, three fixed-point steps.
        took(PATH_EQ35);
        let v = p.ln() + crate::gamma::xsf_lgam(ap1);
        z = ((v + w) / a).exp();
        let mut s = (z / ap1 * (1.0 + z / ap2)).ln_1p();
        z = ((v + z - s) / a).exp();
        s = (z / ap1 * (1.0 + z / ap2)).ln_1p();
        z = ((v + z - s) / a).exp();
        s = (z / ap1 * (1.0 + z / ap2 * (1.0 + z / (a + 3.0)))).ln_1p();
        z = ((v + z - s) / a).exp();
    }
    if z <= 0.01 * ap1 || z > 0.7 * ap1 {
        took(PATH_Z_DIRECT);
        return z;
    }
    // Equation 36.
    took(PATH_EQ36);
    let ls = didonato_sn(a, z).ln();
    let v = p.ln() + crate::gamma::xsf_lgam(ap1);
    z = ((v + z - ls) / a).exp();
    z * (1.0 - (a * z.ln() - z - v + ls) / (a - z))
}

/// Three Halley steps on `P(a, x) = target` (`upper`: `Q(a, x) = target`) from `x`, as
/// `igami`/`igamci` take them: `f/f'` is `(P − p)·x / igam_fac` and `f''/f'` is
/// `(a − 1)/x − 1`, with a Newton step where that ratio overflows, and an early return where
/// `igam_fac` underflows to 0.
fn halley(a: f64, mut x: f64, target: f64, upper: bool) -> f64 {
    for _ in 0..3 {
        let fac = igam_fac(a, x);
        if fac == 0.0 {
            took(PATH_HALLEY_FAC0);
            return x;
        }
        let f_fp = if upper {
            (igamc(a, x) - target) * x / (-fac)
        } else {
            (igam(a, x) - target) * x / fac
        };
        let fpp_fp = -1.0 + (a - 1.0) / x;
        if fpp_fp.is_infinite() {
            took(PATH_NEWTON);
            x -= f_fp;
        } else {
            x -= f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
        }
    }
    x
}

/// The inverse of `P(a, ·)`: the `x` with `P(a, x) = p`, `igami` of `igami.h`, SciPy's
/// `gammaincinv`. `p = 0` is 0 and `p = 1` is inf; above 0.9 it is `igamci(a, 1 − p)`.
///
/// A negative `a` or a `p` outside `[0, 1]` is SciPy's domain error, where xsf sets the error
/// and does NOT return: the solve runs on the invalid input and its NaN is SciPy's answer.
/// The early exits are skipped there, as the C's `else if` chain skips them.
pub(crate) fn igami(a: f64, p: f64) -> f64 {
    if a.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if a >= 0.0 && (0.0..=1.0).contains(&p) {
        if p == 0.0 {
            return 0.0;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }
        if p > 0.9 {
            took(PATH_SWITCH);
            return igamci(a, 1.0 - p);
        }
    }
    let x = find_inverse_gamma(a, p, 1.0 - p);
    halley(a, x, p, false)
}

/// The inverse of `Q(a, ·)`: the `x` with `Q(a, x) = q`, `igamci` of `igami.h`, SciPy's
/// `gammainccinv`. `q = 0` is inf and `q = 1` is 0; above 0.9 it is `igami(a, 1 − q)`. The
/// domain is `igami`'s, with the same fall-through.
pub(crate) fn igamci(a: f64, q: f64) -> f64 {
    if a.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    if a >= 0.0 && (0.0..=1.0).contains(&q) {
        if q == 0.0 {
            return f64::INFINITY;
        }
        if q == 1.0 {
            return 0.0;
        }
        if q > 0.9 {
            took(PATH_SWITCH);
            return igami(a, 1.0 - q);
        }
    }
    let x = find_inverse_gamma(a, 1.0 - q, q);
    halley(a, x, q, true)
}

#[cfg(test)]
mod tests {
    use super::{
        PATH_A1, PATH_CF, PATH_EQ21_EXP, PATH_EQ21_POW, PATH_EQ22, PATH_EQ23, PATH_EQ24,
        PATH_EQ25_LARGE_A, PATH_EQ25_SMALL_A, PATH_EQ31_LARGE_A, PATH_EQ31_UPPER, PATH_EQ33,
        PATH_EQ35, PATH_EQ36, PATH_FAC_LANCZOS, PATH_FAC_LOG1PMX, PATH_FAC_LOGSPACE,
        PATH_FAC_UNDERFLOW, PATH_HALLEY_FAC0, PATH_IGAMC_SERIES, PATH_NEWTON, PATH_SERIES,
        PATH_SWITCH, PATH_TEMME, PATH_Z_DIRECT, TEMME_HITS, ZETA_INT, in_temme_zone, take_paths,
    };
    use crate::convenience::{gammainccinv_scalar, gammaincinv_scalar};
    use crate::gamma::{gammainc_scalar, gammaincc_scalar};
    use fsci_runtime::RuntimeMode;
    use std::hint::black_box;

    fn hits() -> u64 {
        TEMME_HITS.with(std::cell::Cell::get)
    }

    /// `(P, Q)` through the public scalar entry points.
    fn pq(a: f64, x: f64) -> (f64, f64) {
        (
            gammainc_scalar(a, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
            gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
        )
    }

    /// `(a, x, P, Q, P_mp, Q_mp)` for `x ∈ {a, 0.9a, 1.1a, a − 2√a, a + 2√a}` inside the zones,
    /// `x` formed in f64 exactly so. `P` and `Q` are `scipy.special.gammainc`/`gammaincc` from
    /// SciPy 1.17.1; `P_mp` and `Q_mp` are mpmath at 70 digits rounded to f64 (the lower and
    /// the upper integral each, agreeing with `1 − Q` and with a 50-digit run to 1e-50). At
    /// a = 1e10, where `mpmath.gammainc` is too slow, they are 80-digit sums instead: `P` by
    /// its power series and `Q` as the Poisson sum `e^(−x)·Σ_{k<a} xᵏ/k!` (a is an integer),
    /// with `P + Q − 1` below 2e-70 and a 60-digit run agreeing to 2e-50.
    #[rustfmt::skip]
    const ROWS: [(f64, f64, f64, f64, f64, f64); 37] = [
        (25.0, 25.0, 0.5266015314436506, 0.47339846855634937, 0.5266015314436506, 0.47339846855634937),
        (25.0, 22.5, 0.3262068872469913, 0.6737931127530087, 0.3262068872469913, 0.6737931127530087),
        (25.0, 27.500000000000004, 0.7089896993403522, 0.29101030065964784, 0.7089896993403522, 0.29101030065964784),
        (60.0, 60.0, 0.5171692726293873, 0.4828307273706128, 0.5171692726293873, 0.4828307273706128),
        (60.0, 54.0, 0.2240410812709628, 0.7759589187290372, 0.2240410812709628, 0.7759589187290372),
        (60.0, 66.0, 0.7860786212397471, 0.2139213787602529, 0.7860786212397471, 0.21392137876025288),
        (60.0, 44.508066615170335, 0.01539377122768666, 0.9846062287723133, 0.015393771227686686, 0.9846062287723133),
        (60.0, 75.49193338482966, 0.9707604419707545, 0.0292395580292455, 0.9707604419707545, 0.029239558029245525),
        (150.0, 150.0, 0.5108582297493597, 0.4891417702506403, 0.5108582297493597, 0.4891417702506403),
        (150.0, 135.0, 0.10736282957535388, 0.8926371704246461, 0.10736282957535392, 0.8926371704246461),
        (150.0, 165.0, 0.8874634904019882, 0.11253650959801166, 0.8874634904019885, 0.11253650959801158),
        (150.0, 125.50510257216823, 0.018175878445077887, 0.9818241215549222, 0.018175878445077887, 0.9818241215549222),
        (150.0, 174.49489742783177, 0.9730303880547668, 0.026969611945233132, 0.9730303880547668, 0.02696961194523313),
        (199.5, 199.5, 0.5094151950058096, 0.4905848049941904, 0.5094151950058096, 0.4905848049941904),
        (199.5, 179.55, 0.07512148677016707, 0.9248785132298329, 0.07512148677016703, 0.9248785132298329),
        (199.5, 219.45000000000002, 0.9179394294221966, 0.08206057057780343, 0.9179394294221966, 0.08206057057780342),
        (199.5, 171.25110621634894, 0.01880118346745667, 0.9811988165325434, 0.018801183467456656, 0.9811988165325434),
        (199.5, 227.74889378365106, 0.9735686396238532, 0.026431360376146854, 0.9735686396238532, 0.026431360376146813),
        (250.0, 250.0, 0.508410626968991, 0.491589373031009, 0.508410626968991, 0.491589373031009),
        (250.0, 225.0, 0.05305968722805805, 0.9469403127719419, 0.05305968722805805, 0.9469403127719419),
        (250.0, 275.0, 0.9396956137145407, 0.06030438628545924, 0.9396956137145408, 0.060304386285459234),
        (250.0, 218.3772233983162, 0.019233660907914753, 0.9807663390920852, 0.019233660907914778, 0.9807663390920852),
        (250.0, 281.62277660168377, 0.9739475105105316, 0.026052489489468325, 0.9739475105105316, 0.026052489489468336),
        (1e3, 1e3, 0.5042052441802155, 0.4957947558197845, 0.5042052441802155, 0.4957947558197845),
        (1e3, 900.0, 0.0005499022657117818, 0.9994500977342882, 0.0005499022657117829, 0.9994500977342882),
        (1e3, 1100.0, 0.9989406767460701, 0.0010593232539299773, 0.99894067674607, 0.0010593232539299773),
        (1e3, 936.7544467966325, 0.02101650061242861, 0.9789834993875713, 0.02101650061242861, 0.9789834993875713),
        (1e3, 1063.2455532033675, 0.9755701126981652, 0.024429887301834742, 0.9755701126981653, 0.02442988730183475),
        (1e4, 1e4, 0.5013298083399552, 0.4986701916600448, 0.5013298083399552, 0.4986701916600448),
        (1e4, 9800.0, 0.02220754381396969, 0.9777924561860303, 0.022207543813969693, 0.9777924561860303),
        (1e4, 10200.0, 0.9767126778664013, 0.02328732213359879, 0.9767126778664011, 0.023287322133598805),
        (1e6, 1e6, 0.5001329807608725, 0.4998670192391274, 0.5001329807608725, 0.4998670192391274),
        (1e6, 998000.0, 0.022696114006736795, 0.9773038859932632, 0.022696114006736802, 0.9773038859932632),
        (1e6, 1002000.0, 0.9771959041012303, 0.022804095898769836, 0.9771959041012301, 0.022804095898769864),
        (1e10, 1e10, 0.5000013298076014, 0.4999986701923987, 0.5000013298076014, 0.4999986701923987),
        (1e10, 9999800000.0, 0.022749592035814527, 0.9772504079641855, 0.02274959203581455, 0.9772504079641855),
        (1e10, 10000200000.0, 0.9772493281448552, 0.022750671855144743, 0.9772493281448552, 0.02275067185514477),
    ];

    /// Worst relative error of SciPy itself against mpmath over [`ROWS`] is 2.0e-15
    /// (P(1000, 900), 10 ulp); fsci, being bit-for-bit SciPy there, is held to that bound.
    const MPMATH_REL_TOL: f64 = 2.5e-15;

    #[test]
    fn temme_zones_match_scipy_bit_for_bit_and_mpmath() {
        for &(a, x, sp, sq, mp, mq) in &ROWS {
            assert!(in_temme_zone(a, x), "row ({a}, {x}) is not in a Temme zone");
            let before = hits();
            let (p, q) = pq(a, x);
            assert_eq!(
                hits(),
                before + 2,
                "({a}, {x}) did not take the Temme branch"
            );
            // Bit-for-bit, with libm's `exp` (inside `erfc` and the prefactor) as the only
            // shared dependency: SciPy's wheel and fsci both call the host's.
            assert_eq!(
                p.to_bits(),
                sp.to_bits(),
                "gammainc({a}, {x}) = {p:e}, SciPy {sp:e}"
            );
            assert_eq!(
                q.to_bits(),
                sq.to_bits(),
                "gammaincc({a}, {x}) = {q:e}, SciPy {sq:e}"
            );
            let rel_p = ((p - mp) / mp).abs();
            let rel_q = ((q - mq) / mq).abs();
            assert!(
                rel_p <= MPMATH_REL_TOL && rel_q <= MPMATH_REL_TOL,
                "({a}, {x}): P rel {rel_p:e}, Q rel {rel_q:e} against mpmath"
            );
        }
    }

    #[test]
    fn temme_branch_fires_inside_the_zones_and_nowhere_else() {
        // Just inside each edge of both zones: a above 20, below and above 200, and
        // |x − a|/a just under 0.3 (zone 1) and just under 4.5/√a = 0.045 at a = 1e4 (zone 2).
        let inside = [
            (20.000001, 20.000001),
            (25.0, 25.0 * 1.29999),
            (25.0, 25.0 * 0.70001),
            (199.99999, 199.99999),
            (200.00001, 200.00001),
            (1e4, 1e4 * (1.0 + 0.04499)),
            (1e4, 1e4 * (1.0 - 0.04499)),
            (1e10, 1e10),
        ];
        // Just outside the same edges, a = 20 and a = 200 themselves (neither zone), rows of
        // the reference sweep that fall outside, and every special case.
        let outside = [
            (19.9, 19.9),
            (20.0, 20.0),
            (200.0, 200.0),
            (25.0, 25.0 * 1.30001),
            (25.0, 25.0 * 0.69999),
            (1e4, 1e4 * (1.0 + 0.04501)),
            (1e4, 1e4 * (1.0 - 0.04501)),
            (25.0, 15.0),
            (1e4, 9000.0),
            (1e10, 9e9),
            (5.0, 5.0),
            (25.0, 0.0),
            (25.0, f64::INFINITY),
            (f64::INFINITY, 25.0),
            (f64::NAN, 25.0),
            (25.0, f64::NAN),
        ];
        for (a, x) in inside {
            assert!(in_temme_zone(a, x), "({a}, {x}) should be in a zone");
            let before = hits();
            let (p, q) = pq(a, x);
            assert_eq!(hits(), before + 2, "({a}, {x}) missed the Temme branch");
            assert!(
                p.is_finite() && q.is_finite() && (p + q - 1.0).abs() <= 1e-15,
                "({a}, {x}): P = {p:e}, Q = {q:e}"
            );
        }
        for (a, x) in outside {
            assert!(
                !in_temme_zone(a, x),
                "({a}, {x}) should be outside the zones"
            );
            let before = hits();
            let _ = pq(a, x);
            assert_eq!(hits(), before, "({a}, {x}) took the Temme branch");
        }
    }

    /// Bit-for-bit equality, with every NaN equal to every NaN (SciPy's NaN payload is not
    /// part of its contract).
    fn same_bits(got: f64, want: f64) -> bool {
        (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
    }

    /// `(a, x, P, P's branches, Q, Q's branches)` (frankenscipy-449uv). `P` and `Q` are SciPy
    /// 1.17.1's `gammainc`/`gammaincc`; the branch sets are the exact set of `igam.h` paths a
    /// Python emulation of xsf 0d0a593f takes there, and that emulation returns SciPy's bits at
    /// every row. Every branch appears: the power series, the continued fraction,
    /// `igamc_series`, both Temme zones, all four `igam_fac` forms (the log-space `exp`, its
    /// underflow to 0, and the Lanczos form through `pow` and through `log1pmx`), tails down
    /// to 6e-263, `a` subnormal (xsf's `lgam` is +inf there, so `P = 0`), `P > 1` at
    /// `a = 1e-300`, and every edge `igam` and `igamc` special-case. The old kernel answered NaN
    /// at `a = 0` and at an infinite `a`, and was off by ulps on most interior rows.
    #[rustfmt::skip]
    const FORWARD: [(f64, f64, f64, u32, f64, u32); 28] = [
        (0.5, 0.3, 0.5614219739190003, PATH_FAC_LANCZOS | PATH_SERIES, 0.4385780260809997, PATH_FAC_LANCZOS | PATH_SERIES),
        (3.0, 1.5, 0.19115316946194183, PATH_FAC_LOGSPACE | PATH_SERIES, 0.8088468305380582, PATH_FAC_LOGSPACE | PATH_SERIES),
        (2.0, 10.0, 0.9995006007726127, PATH_CF | PATH_FAC_LOGSPACE, 0.0004993992273873336, PATH_CF | PATH_FAC_LOGSPACE),
        (0.1, 0.2, 0.8794196267900569, PATH_FAC_LOGSPACE | PATH_SERIES, 0.12058037320994318, PATH_IGAMC_SERIES),
        (0.01, 0.9, 0.9973736567479485, PATH_FAC_LOGSPACE | PATH_SERIES, 0.0026263432520511505, PATH_IGAMC_SERIES),
        (0.5, 1.05, 0.8527008613773283, PATH_IGAMC_SERIES, 0.14729913862267172, PATH_IGAMC_SERIES),
        (50.0, 45.0, 0.24680203440017026, PATH_TEMME, 0.7531979655998298, PATH_TEMME),
        (10000.0, 10100.0, 0.8413487504471796, PATH_TEMME, 0.15865124955282037, PATH_TEMME),
        (300.0, 400.0, 0.9999999249261916, PATH_CF | PATH_FAC_LOG1PMX, 7.507380835521643e-08, PATH_CF | PATH_FAC_LOG1PMX),
        (10.0, 12.0, 0.7576078383294875, PATH_CF | PATH_FAC_LANCZOS, 0.24239216167051245, PATH_CF | PATH_FAC_LANCZOS),
        (1000.0, 1.0, 0.0, PATH_FAC_UNDERFLOW | PATH_SERIES, 1.0, PATH_FAC_UNDERFLOW | PATH_SERIES),
        (1.0, 800.0, 1.0, PATH_CF | PATH_FAC_UNDERFLOW, 0.0, PATH_CF | PATH_FAC_UNDERFLOW),
        (0.5, 600.0, 1.0, PATH_CF | PATH_FAC_LOGSPACE, 6.099568814808675e-263, PATH_CF | PATH_FAC_LOGSPACE),
        (250.0, 20.0, 1.2533481488424743e-176, PATH_FAC_LOGSPACE | PATH_SERIES, 1.0, PATH_FAC_LOGSPACE | PATH_SERIES),
        (1e-310, 0.5, 0.0, PATH_FAC_UNDERFLOW | PATH_SERIES, 1.1593151565844e-311, PATH_IGAMC_SERIES),
        (1e-300, 0.5, 1.0000000000000233, PATH_FAC_LOGSPACE | PATH_SERIES, 5.597735947761714e-301, PATH_IGAMC_SERIES),
        (0.0, 1.0, 1.0, 0, 0.0, 0),
        (-0.0, 1.0, 1.0, 0, 0.0, 0),
        (0.0, 0.0, f64::NAN, 0, f64::NAN, 0),
        (0.0, f64::INFINITY, 1.0, 0, 0.0, 0),
        (f64::INFINITY, 1.0, 0.0, 0, 1.0, 0),
        (f64::INFINITY, f64::INFINITY, f64::NAN, 0, f64::NAN, 0),
        (1.0, f64::INFINITY, 1.0, 0, 0.0, 0),
        (2.0, 0.0, 0.0, 0, 1.0, 0),
        (-1.0, 1.0, f64::NAN, 0, f64::NAN, 0),
        (1.0, -1.0, f64::NAN, 0, f64::NAN, 0),
        (f64::NAN, 1.0, f64::NAN, 0, f64::NAN, 0),
        (1.0, f64::NAN, f64::NAN, 0, f64::NAN, 0),
    ];

    #[test]
    fn gammainc_gammaincc_are_scipy_bit_for_bit_on_every_branch() {
        for &(a, x, p, p_paths, q, q_paths) in &FORWARD {
            take_paths();
            let got_p = gammainc_scalar(black_box(a), black_box(x), RuntimeMode::Strict)
                .expect("Strict gammainc answers every input");
            let took_p = take_paths();
            let got_q = gammaincc_scalar(black_box(a), black_box(x), RuntimeMode::Strict)
                .expect("Strict gammaincc answers every input");
            let took_q = take_paths();
            assert!(
                same_bits(got_p, p),
                "gammainc({a:e}, {x:e}) = {got_p:e}, SciPy {p:e}"
            );
            assert!(
                same_bits(got_q, q),
                "gammaincc({a:e}, {x:e}) = {got_q:e}, SciPy {q:e}"
            );
            assert_eq!(took_p, p_paths, "gammainc({a:e}, {x:e}) took {took_p:#b}");
            assert_eq!(took_q, q_paths, "gammaincc({a:e}, {x:e}) took {took_q:#b}");
        }
    }

    /// `(a, p, x, branches)` for `gammaincinv` (SciPy 1.17.1's value; branch sets as in
    /// [`FORWARD`]). Every estimate of `find_inverse_gamma` appears (`a = 1` both ways,
    /// DiDonato & Morris 21 in both forms, 22 to 25, 31 in both of its direct returns, 33, 35,
    /// 36 and the direct `z`), the Halley loop's early return and its Newton fallback, the
    /// switch to `igamci` above p = 0.9, and the domain errors, whose NaN comes out of the
    /// solve itself as in the C.
    #[rustfmt::skip]
    const INVERSE_P: [(f64, f64, f64, u32); 26] = [
        (1.0, 0.1, 0.10536051565782636, PATH_A1 | PATH_FAC_LOGSPACE | PATH_SERIES),
        (1.0, 1e-300, 1e-300, PATH_A1 | PATH_FAC_LOGSPACE | PATH_SERIES),
        (0.001, 1e-300, 0.0, PATH_EQ21_POW | PATH_FAC_UNDERFLOW | PATH_HALLEY_FAC0),
        (0.1, 0.8, 0.06938988323997317, PATH_EQ21_POW | PATH_FAC_LANCZOS | PATH_SERIES),
        (1.5, 0.6, 1.473083036550975, PATH_EQ31_UPPER | PATH_FAC_LANCZOS | PATH_SERIES),
        (1.5, 0.01, 0.057415900949558535, PATH_EQ35 | PATH_EQ36 | PATH_FAC_LOGSPACE | PATH_SERIES),
        (1.5, 1e-300, 1.2089939655123954e-200, PATH_EQ35 | PATH_FAC_LOGSPACE | PATH_SERIES | PATH_Z_DIRECT),
        (30.0, 0.05, 21.593979226994882, PATH_EQ36 | PATH_FAC_LANCZOS | PATH_TEMME),
        (300.0, 1e-10, 202.61040319416256, PATH_EQ36 | PATH_FAC_LOG1PMX | PATH_SERIES),
        (1000000.0, 0.5, 999999.6666666864, PATH_EQ31_LARGE_A | PATH_FAC_LOG1PMX | PATH_TEMME),
        (0.9, 1e-279, 9.5760886990836e-311, PATH_EQ21_POW | PATH_FAC_LOGSPACE | PATH_NEWTON | PATH_SERIES),
        (0.001, 0.95, 2.973587549646711e-23, PATH_EQ21_POW | PATH_FAC_LOGSPACE | PATH_IGAMC_SERIES | PATH_SWITCH),
        (3.0, 0.999, 11.228872242412661, PATH_CF | PATH_EQ33 | PATH_FAC_LOGSPACE | PATH_SWITCH),
        (1.5, 0.99, 5.6724333650721865, PATH_CF | PATH_EQ25_LARGE_A | PATH_FAC_LOGSPACE | PATH_SWITCH),
        (0.001, 0.999999, 5.120025083764957, PATH_CF | PATH_EQ25_SMALL_A | PATH_FAC_LOGSPACE | PATH_SWITCH),
        (0.1, 0.95, 0.5804351053231342, PATH_EQ22 | PATH_FAC_LOGSPACE | PATH_IGAMC_SERIES | PATH_SWITCH),
        (0.05, 0.99, 1.0876274000918094, PATH_CF | PATH_EQ23 | PATH_FAC_LOGSPACE | PATH_IGAMC_SERIES | PATH_SWITCH),
        (2.0, 0.0, 0.0, 0),
        (2.0, 1.0, f64::INFINITY, 0),
        (0.0, 0.5, f64::NAN, PATH_EQ21_POW | PATH_FAC_LOG1PMX),
        (f64::INFINITY, 0.5, f64::NAN, PATH_FAC_LOG1PMX | PATH_Z_DIRECT),
        (-1.0, 0.5, f64::NAN, PATH_EQ25_SMALL_A | PATH_FAC_LOG1PMX),
        (2.0, 1.5, f64::NAN, PATH_EQ33 | PATH_FAC_LOG1PMX),
        (2.0, -0.1, f64::NAN, PATH_EQ36 | PATH_FAC_LOG1PMX),
        (f64::NAN, 0.5, f64::NAN, 0),
        (-0.0, 0.0, 0.0, 0),
    ];

    /// `(a, q, x, branches)` for `gammainccinv`, as [`INVERSE_P`]: DiDonato & Morris 21's
    /// exponential form and 24, the upper tail from q = 1e-300 (25 for `a < 1` and `a > 1`,
    /// 33, `a = 1`), the switch to `igami` above q = 0.9, and the edges.
    #[rustfmt::skip]
    const INVERSE_Q: [(f64, f64, f64, u32); 19] = [
        (1e-06, 1e-05, 2.548961730206411e-05, PATH_EQ21_EXP | PATH_FAC_LOGSPACE | PATH_IGAMC_SERIES),
        (0.1, 0.012, 1.4609628189761006, PATH_CF | PATH_EQ24 | PATH_FAC_LOGSPACE),
        (0.001, 0.6, 0.0, PATH_EQ21_POW | PATH_FAC_UNDERFLOW | PATH_HALLEY_FAC0),
        (3.0, 1e-06, 19.129168188604844, PATH_CF | PATH_EQ33 | PATH_FAC_LOGSPACE),
        (1.5, 1e-300, 694.1683869273429, PATH_CF | PATH_EQ25_LARGE_A | PATH_FAC_LOGSPACE),
        (0.001, 1e-300, 677.3551998031188, PATH_CF | PATH_EQ25_SMALL_A | PATH_FAC_LOGSPACE),
        (1.0, 1e-300, 690.7755278982137, PATH_A1 | PATH_CF | PATH_FAC_LOGSPACE),
        (1.0, 0.95, 0.05129329438755058, PATH_A1 | PATH_FAC_LOGSPACE | PATH_SERIES | PATH_SWITCH),
        (1.5, 0.999, 0.012148792907846375, PATH_EQ35 | PATH_FAC_LOGSPACE | PATH_SERIES | PATH_SWITCH | PATH_Z_DIRECT),
        (30.0, 0.1, 37.1985028596843, PATH_EQ31_UPPER | PATH_FAC_LANCZOS | PATH_TEMME),
        (300.0, 1e-06, 389.64230255489844, PATH_CF | PATH_EQ31_UPPER | PATH_FAC_LOG1PMX),
        (0.5, 1e-200, 456.88135039294025, PATH_CF | PATH_EQ23 | PATH_FAC_LOGSPACE),
        (2.0, 0.0, f64::INFINITY, 0),
        (2.0, 1.0, 0.0, 0),
        (0.0, 0.5, f64::NAN, PATH_EQ21_POW | PATH_FAC_LOG1PMX),
        (f64::INFINITY, 0.5, f64::NAN, PATH_FAC_LOG1PMX | PATH_Z_DIRECT),
        (-1.0, 0.5, f64::NAN, PATH_EQ25_SMALL_A | PATH_FAC_LOG1PMX),
        (2.0, 1.5, f64::NAN, PATH_EQ36 | PATH_FAC_LOG1PMX),
        (0.5, f64::NAN, f64::NAN, 0),
    ];

    #[test]
    fn gammaincinv_gammainccinv_are_scipy_bit_for_bit_on_every_branch() {
        for &(a, p, x, paths) in &INVERSE_P {
            take_paths();
            let got = gammaincinv_scalar(black_box(a), black_box(p));
            let took = take_paths();
            assert!(
                same_bits(got, x),
                "gammaincinv({a:e}, {p:e}) = {got:e}, SciPy {x:e}"
            );
            assert_eq!(took, paths, "gammaincinv({a:e}, {p:e}) took {took:#b}");
        }
        for &(a, q, x, paths) in &INVERSE_Q {
            take_paths();
            let got = gammainccinv_scalar(black_box(a), black_box(q));
            let took = take_paths();
            assert!(
                same_bits(got, x),
                "gammainccinv({a:e}, {q:e}) = {got:e}, SciPy {x:e}"
            );
            assert_eq!(took, paths, "gammainccinv({a:e}, {q:e}) took {took:#b}");
        }
    }

    /// SciPy 1.17.1's value of each distribution wrapper: `chdtri` and `pdtri` from p = 1e-300
    /// to 1 − 2^-53 and `gdtrix`/`gdtria` from 1e-300 to 1 − 2^-53 (the tails where a hard-region
    /// sweep found the old kernels 6e-2 to 1e163 off, frankenscipy-449uv), `pdtri`'s C `int`
    /// cast (truncation, NaN past `INT_MAX` and at it, where `k + 1` wraps), the forward CDFs
    /// on each `igam`/`igamc` branch, and every edge where SciPy's wrappers differ from a
    /// plain domain check (`chdtr(0, 1) = 1`, `pdtr(NaN, 0) = 1`, `gdtr(1, 0, 3) = 1`,
    /// `gdtrix(inf, 2, 0.5)` NaN, `gdtria(0.5, 2, inf) = 0`).
    #[rustfmt::skip]
    const WRAPPERS: &[(&str, &[f64], f64)] = &[
        ("chdtri", &[0.5, 1e-300], 1369.1795943036545),
        ("chdtri", &[0.5, 1e-200], 909.2754513962253),
        ("chdtri", &[0.5, 1e-100], 449.81082449021244),
        ("chdtri", &[0.5, 1e-10], 38.94975469094148),
        ("chdtri", &[0.5, 0.3], 0.3746964567403943),
        ("chdtri", &[0.5, 0.9999999999999999], 2.050950835438566e-64),
        ("chdtri", &[3.0, 1e-300], 1388.3367738546858),
        ("chdtri", &[3.0, 1e-200], 927.4170108566861),
        ("chdtri", &[3.0, 1e-100], 466.2143575712913),
        ("chdtri", &[3.0, 1e-10], 49.542155927523666),
        ("chdtri", &[3.0, 0.3], 3.6648707831703162),
        ("chdtri", &[3.0, 0.9999999999999999], 5.5854857570344605e-11),
        ("chdtri", &[20.0, 1e-300], 1474.8286249113887),
        ("chdtri", &[20.0, 1e-200], 1007.46314903344),
        ("chdtri", &[20.0, 1e-100], 535.6059952832642),
        ("chdtri", &[20.0, 1e-10], 89.25571443411813),
        ("chdtri", &[20.0, 0.3], 22.774545073646436),
        ("chdtri", &[20.0, 0.9999999999999999], 0.23234424623235955),
        ("chdtri", &[0.0, 1.0], 0.0),
        ("chdtri", &[0.0, 0.0], f64::INFINITY),
        ("chdtri", &[3.0, 1.5], f64::NAN),
        ("pdtri", &[0.0, 1e-300], 690.7755278982137),
        ("pdtri", &[0.0, 1e-100], 230.25850929940455),
        ("pdtri", &[0.0, 1e-10], 23.025850929940457),
        ("pdtri", &[0.0, 0.3], 1.2039728043259357),
        ("pdtri", &[0.0, 0.999], 0.0010005003335835346),
        ("pdtri", &[5.0, 1e-300], 718.8835024894933),
        ("pdtri", &[5.0, 1e-100], 253.161013464717),
        ("pdtri", &[5.0, 1e-10], 36.347197026820986),
        ("pdtri", &[5.0, 0.3], 7.005550084210963),
        ("pdtri", &[5.0, 0.999], 1.1071046602556396),
        ("pdtri", &[30.0, 1e-300], 817.3361521675091),
        ("pdtri", &[30.0, 1e-100], 329.6349358303983),
        ("pdtri", &[30.0, 1e-10], 80.45360804958545),
        ("pdtri", &[30.0, 0.3], 33.66100572955601),
        ("pdtri", &[30.0, 0.999], 16.59053710164774),
        ("pdtri", &[2.7, 1e-300], 703.1964976004614),
        ("pdtri", &[2.7, 1e-100], 240.5394448440991),
        ("pdtri", &[2.7, 1e-10], 29.14590147882952),
        ("pdtri", &[2.7, 0.3], 3.615567665865991),
        ("pdtri", &[2.7, 0.999], 0.19053337756840327),
        ("pdtri", &[-0.5, 0.5], 0.6931471805599455),
        ("pdtri", &[-1.0, 0.5], f64::NAN),
        ("pdtri", &[2147483646.0, 0.5], 2147483646.6666667),
        ("pdtri", &[2147483647.0, 0.5], f64::NAN),
        ("pdtri", &[3000000000.0, 0.5], f64::NAN),
        ("pdtri", &[f64::NAN, 0.5], f64::NAN),
        ("pdtri", &[2.0, 0.0], f64::INFINITY),
        ("pdtri", &[2.0, 1.0], f64::NAN),
        ("gdtrix", &[0.1, 0.5, 1e-300], 0.0),
        ("gdtria", &[1e-300, 0.5, 0.1], 0.0),
        ("gdtrix", &[0.1, 0.5, 1e-150], 7.853981633974481e-300),
        ("gdtria", &[1e-150, 0.5, 0.1], 7.853981633974481e-300),
        ("gdtrix", &[0.1, 0.5, 1e-20], 7.85398163397449e-40),
        ("gdtria", &[1e-20, 0.5, 0.1], 7.85398163397449e-40),
        ("gdtrix", &[0.1, 0.5, 0.4], 1.3749794886422795),
        ("gdtria", &[0.4, 0.5, 0.1], 1.3749794886422795),
        ("gdtrix", &[0.1, 0.5, 0.999], 54.13783085331366),
        ("gdtria", &[0.999, 0.5, 0.1], 54.13783085331366),
        ("gdtrix", &[0.1, 0.5, 0.9999999999], 209.10728101491392),
        ("gdtria", &[0.9999999999, 0.5, 0.1], 209.10728101491392),
        ("gdtrix", &[0.1, 0.5, 0.9999999999999999], 343.816261058342),
        ("gdtria", &[0.9999999999999999, 0.5, 0.1], 343.816261058342),
        ("gdtrix", &[2.5, 20.0, 1e-300], 3.321744481495734e-15),
        ("gdtria", &[1e-300, 20.0, 2.5], 3.321744481495734e-15),
        ("gdtrix", &[2.5, 20.0, 1e-150], 1.0504278497978574e-07),
        ("gdtria", &[1e-150, 20.0, 2.5], 1.0504278497978574e-07),
        ("gdtrix", &[2.5, 20.0, 1e-20], 0.3461344461004619),
        ("gdtria", &[1e-20, 20.0, 2.5], 0.3461344461004619),
        ("gdtrix", &[2.5, 20.0, 0.4], 7.426791895816928),
        ("gdtria", &[0.4, 20.0, 2.5], 7.426791895816928),
        ("gdtrix", &[2.5, 20.0, 0.999], 14.680391503798205),
        ("gdtria", &[0.999, 20.0, 2.5], 14.680391503798205),
        ("gdtrix", &[2.5, 20.0, 0.9999999999], 25.060965584686592),
        ("gdtria", &[0.9999999999, 20.0, 2.5], 25.060965584686592),
        ("gdtrix", &[2.5, 20.0, 0.9999999999999999], 32.48008259081265),
        ("gdtria", &[0.9999999999999999, 20.0, 2.5], 32.48008259081265),
        ("gdtrix", &[f64::INFINITY, 2.0, 0.5], f64::NAN),
        ("gdtrix", &[0.0, 0.0, 0.5], f64::NAN),
        ("gdtrix", &[1.0, 0.0, 0.0], 0.0),
        ("gdtria", &[0.0, 0.0, 2.0], 0.0),
        ("gdtria", &[0.0, 0.0, f64::INFINITY], f64::NAN),
        ("gdtria", &[0.5, 2.0, f64::INFINITY], 0.0),
        ("chdtr", &[1.0, 0.6], 0.5614219739190003),
        ("chdtrc", &[1.0, 0.6], 0.4385780260809997),
        ("pdtr", &[0.5, 0.3], 0.740818220681718),
        ("pdtrc", &[0.5, 0.3], 0.25918177931828207),
        ("gdtr", &[0.5, 0.5, 0.6], 0.5614219739190003),
        ("gdtrc", &[0.5, 0.5, 0.6], 0.4385780260809997),
        ("chdtr", &[4.0, 20.0], 0.9995006007726127),
        ("chdtrc", &[4.0, 20.0], 0.0004993992273873336),
        ("pdtr", &[2.0, 10.0], 0.0027693957155115775),
        ("pdtrc", &[2.0, 10.0], 0.9972306042844884),
        ("gdtr", &[0.5, 2.0, 20.0], 0.9995006007726127),
        ("gdtrc", &[0.5, 2.0, 20.0], 0.0004993992273873336),
        ("chdtr", &[100.0, 90.0], 0.24680203440017026),
        ("chdtrc", &[100.0, 90.0], 0.7531979655998298),
        ("pdtr", &[50.0, 45.0], 0.7962802904074182),
        ("pdtrc", &[50.0, 45.0], 0.2037197095925817),
        ("gdtr", &[0.5, 50.0, 90.0], 0.24680203440017026),
        ("gdtrc", &[0.5, 50.0, 90.0], 0.7531979655998298),
        ("chdtr", &[600.0, 800.0], 0.9999999249261916),
        ("chdtrc", &[600.0, 800.0], 7.507380835521643e-08),
        ("pdtr", &[300.0, 400.0], 1.0103960151958101e-07),
        ("pdtrc", &[300.0, 400.0], 0.9999998989603984),
        ("gdtr", &[0.5, 300.0, 800.0], 0.9999999249261916),
        ("gdtrc", &[0.5, 300.0, 800.0], 7.507380835521643e-08),
        ("chdtr", &[0.0, 1.0], 1.0),
        ("chdtr", &[1.0, -1.0], f64::NAN),
        ("chdtr", &[-1.0, 1.0], f64::NAN),
        ("chdtrc", &[0.0, 1.0], 0.0),
        ("pdtr", &[f64::NAN, 0.0], 1.0),
        ("pdtrc", &[f64::NAN, 0.0], 0.0),
        ("pdtr", &[2.7, 3.0], 0.42319008112684364),
        ("pdtr", &[-0.5, 1.0], f64::NAN),
        ("gdtr", &[0.0, 2.0, 3.0], 0.0),
        ("gdtr", &[1.0, 0.0, 3.0], 1.0),
        ("gdtr", &[1.0, 2.0, -1.0], f64::NAN),
        ("gdtrc", &[1.0, 2.0, -1.0], f64::NAN),
    ];

    #[test]
    fn distribution_wrappers_are_scipy_bit_for_bit_in_the_tails_and_at_the_edges()
    -> Result<(), String> {
        use crate::gamma::{
            chdtr, chdtrc, chdtri, gdtr, gdtrc, gdtria, gdtrix, pdtr, pdtrc, pdtri,
        };
        for &(name, args, want) in WRAPPERS {
            let v: Vec<f64> = args.iter().map(|&t| black_box(t)).collect();
            let got = match name {
                "chdtr" => chdtr(v[0], v[1]),
                "chdtrc" => chdtrc(v[0], v[1]),
                "chdtri" => chdtri(v[0], v[1]),
                "pdtr" => pdtr(v[0], v[1]),
                "pdtrc" => pdtrc(v[0], v[1]),
                "pdtri" => pdtri(v[0], v[1]),
                "gdtr" => gdtr(v[0], v[1], v[2]),
                "gdtrc" => gdtrc(v[0], v[1], v[2]),
                "gdtria" => gdtria(v[0], v[1], v[2]),
                "gdtrix" => gdtrix(v[0], v[1], v[2]),
                other => return Err(format!("no wrapper {other}")),
            };
            assert!(
                same_bits(got, want),
                "{name}{args:?} = {got:e}, SciPy {want:e}"
            );
        }
        Ok(())
    }

    /// Every entry of [`ZETA_INT`] is what Cephes' `zeta(n, 1)` returns, and the table is
    /// not simply `ζ(n)` rounded: at n = 2 Cephes is one ulp above `π²/6`.
    #[test]
    fn zeta_int_is_cephes_zeta_bit_for_bit() {
        for (k, &z) in ZETA_INT.iter().enumerate() {
            let n = (k + 2) as f64;
            let cephes = crate::convenience::hurwitz_zeta(black_box(n), black_box(1.0));
            assert_eq!(
                z.to_bits(),
                cephes.to_bits(),
                "zeta({n}, 1): table {z:e}, Cephes {cephes:e}"
            );
        }
        let nearest = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert_eq!(ZETA_INT[0].to_bits(), nearest.to_bits() + 1);
    }
}
