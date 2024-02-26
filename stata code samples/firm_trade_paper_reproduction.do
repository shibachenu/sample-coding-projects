clear all

/*
1.2. MeasuringandExplainingManagementPracticesAcrossFirmsandCountries. BloomandVanReenen (2007) conduct a survey to score 732 firms across France, Germany, the U.K., and the U.S. on 18 management practices. (See Appendix I.A for the full list and example questions used in the survey.) They
then score each firm on their performance in these 18 categories, using these scores as a quantification
of management quality. In this problem, you will replicate and discuss some of their empirical findings. The necessary data and an accompanying guide for variable names can be found on Canvas.
(1) Recreate the histograms in Figure I, showing the distribution of management scores by country. Interpret these plots.
*/

* Import dataset
use "Data and starter code/realdata.dta", clear
* Histograms for France, UK, Germany, US
use "Data and starter code/realdata.dta" if ccfrance|ccuk|ccgermany|ccus, clear
histogram amanagement, by(cty, title("Distribution of Management Scores by Country")) bin(50) ylabel(0 0.2 0.4 0.6 0.8 1.0 1.2)


*(2) Recreate the two histograms in Figure II, plotting the distribution of management scores for (i) all firms in the sample reporting low competition (competition<=2) and/or primogeniture family succession, and (ii) all firms reporting high-competition (competition==3) and no primo- geniture succession. (Be sure to drop observations for which there is missing ownership or CEO data, famceomiss==1.) Report the means and standard deviations for both samples. What does this evidence suggest? Provide some intuition for the result. Is this evidence causal? Explain.
bysort code cty: gen dup = cond(_N==1,0,_n) 
drop if famceomiss==1|famcontrolmiss==1
drop if dup>1

histogram amanagement if competition<=2|famcontrol50_2ceop==1, title("Low competition and primogeniture family firms") bin(50) ylabel(0 0.2 0.4 0.6 0.8 1.0 1.2)
histogram amanagement if competition==3&famcontrol50_2ceop==0, title("High competition and non-primogeniture family firms") bin(50) ylabel(0 0.2 0.4 0.6 0.8 1.0 1.2)

*Means and std dev for both samples
sum amanagement if competition<=2|famcontrol50_2ceop==1
sum amanagement if competition==3&famcontrol50_2ceop==0


/*(3) Bloom and Van Reenen (2007) consider the basic firm production function:
yc =αclc +αckc +αcnc +βcMc+γcZc +uc it lit kit nit i it it
where Y is deflated sales, L is labor, K is capital, N is intermediate inputs (materials), and M is management practices of firm i in country c at time t. Lower case letters denote natural logs (y = ln(Y), etc.). Inside Z consists a number of other controls that affect productivity, such as workforce characteristics, firm characteristics, and a complete set of three-digit industry dum- mies and country dummies.
You will now replicate the first four columns of Table I to investigate the association between firm performance and management practices. For all regressions, report heteroskedasticity- robust standard errors clustered at the firm level.*/

*column 1: regress on management z score, log Labor, control for country, time, and industry dummies
reg ls zmanagement le le_fr le_gr le_us le_uncons uncons cyy* if le~=. & lp~=.& lm~=., vce(cluster code)
*column 2: regress on management z score, log Labor, log capital, materials, control for country, time, and industry dummies
reg ls zmanagement le lp lm le_fr le_gr le_us le_uncons lp_fr lp_gr lp_us lp_uncons lm_fr lm_gr lm_us lm_un uncons cyy* if le~=. & lp~=.& lm~=., vce(cluster code)
*column 3: regress on management z score, log Labor, log capital, materials, control for country, time, and industry dummies with more general control
reg ls zmanagement le lp lm le_fr le_gr le_us le_uncons lp_fr lp_gr lp_us lp_uncons lm_fr lm_gr lm_us lm_un uncons cyy* lfirmage public ldegree ldegreemiss mba mbamiss lhrs i.sic3 if le~=. & lp~=.& lm~=., vce(cluster code)
*column 4: regress on management z score, log Labor, log capital, materials, control for country, time, and industry dummies with more general control, and noise control
reg ls zmanagement le lp lm le_fr le_gr le_us le_uncons lp_fr lp_gr lp_us lp_uncons lm_fr lm_gr lm_us lm_un uncons cyy* lfirmage public ldegree ldegreemiss mba mbamiss lhrs gender sen1 tenurepost countries day2 day3 day4 day5 timelocal duration reli aa* i.sic3 if le~=. & lp~=.& lm~=., vce(cluster code)

*Write up for questions in the main doc


****************************************************
* Code for for Problem Set 1 - problem 2 in 14.76, Spring 2024 *
* Data is from Cai and Szeidl (2017)               *
****************************************************

* Setup -- change pset_folder to your folder!
clear all
global pset_folder "/Users/ifwonderland/Library/CloudStorage/GoogleDrive-ifwonderland@gmail.com/My Drive/MIT DEDP/Courses/14.760 Firms, Markets, Trade, Growth/pset1"
global data_folder "${pset_folder}/Data and starter code"

* Variable Definitions
* num_employee - The number of employees
* after1 - An indicator that is equal to one if the round is the midline survey
* after2 - An indicator that is equal to one if the round is the endline survey
* treatment - An indicator that is equal to one if the 
* firmid - A unique identifier for each firm
* clusterid - The meeting group identifier (or firm identifier for control firms), which is also the level at which the standard errors should be clustered


* Load Data
use "${data_folder}/120617main.dta", clear
gen post=0
replace post=1 if round>1
gen intervention1=treatment*after1
gen intervention2=treatment*after2
gen intervention=treatment*post

*where Yit is an outcome for firm i in survey round t, γi is a firm fixed effect, Treatedi is an indicator for being part of the treated group, Midlinet and Endlinet are indicators equal to one when the survey round is the midline or endline, respectively, and εit is the residual. They generate standard errors clustering at the meeting group level for treated firms, and at the firm level for control firms.

* Estimate Treatment Effects for Number of Employees
*XX - Each Place Where I have written "XX," is somewhere that you need to write your own code! Good Luck!

*Num of employees
reghdfe num_employee after1 after2 intervention1 intervention2, absorb(firmid) vce(cluster clusterid)
*ln num of employees for cross check with paper result
*reghdfe lnnum_employee after1 after2 intervention1 intervention2, absorb(firmid) vce(cluster clusterid)

* Estimate Treatment Effect Without Individual Fixed Effects
reghdfe num_employee after1 after2 intervention1 intervention2, vce(cluster clusterid)

* Go Back to Estimating Everything with Individual Fixed Effects
* Pool Midline*Treated and Endline*Treated into just Post*Treated
reghdfe num_employee after1 after2 intervention, absorb(firmid) vce(cluster clusterid)


/*
(9) Required only for 14.760 students, bonus credit for 14.76 students. We will estimate a distri- bution regression: that is, we will estimate the effect of the intervention on different parts of the distribution, then plot the results. To do this, for each value x from 1 to 100:
(a) Create an indicator variable that is equal to one when the number of employees is greater than or equal to x.
(b) Run the regression from (f), but with the indicator variable you constructed above as the outcome.
(c) Recordtheestimatedcoefficientandstandarderror(Note:InStata,youcanaccesstheestimated coefficient on the variable intervention with _b[intervention] , and the standard error with _se[intervention] ).
(d) Use these estimates to generate the confidence intervals for each estimate.
(e) Plot the point estimates and confidence intervals over x.
In the starter code, we have given you the for-loop and the code to make the graph, but you have to fill in what's inside the loop, as well as generate the confidence intervals.
In your solutions, report the coefficient and standard error for x = 2, x = 5, x = 10, and x = 50. Also, include the plot of your results. On what part(s) of the distribution is the effect concentrated?
*/
*** Do Distribution Effects ***
gen coeff = .
gen std_error = .
gen x = _n
forvalues i = 1/100 {
	gen firm_size_`i'more = 0
	replace firm_size_`i'more = 1 if num_employee >= `i'
	qui reghdfe firm_size_`i'more after1 after2 intervention, absorb(firmid) vce(cluster clusterid)
	replace coeff =  _b[intervention] if x==`i'
	replace std_error = _se[intervention] if x==`i'
	drop firm_size_`i'more
}

gen upper_ci = coeff + invnorm(0.975)*std_error
gen lower_ci = coeff - invnorm(0.975)*std_error
list x coeff std_error if inlist(x,2,5,10,50)
tw (connected coeff x) (line upper_ci lower_ci x, lpattern(dash dash) lcolor(gray gray)) if !mi(coeff) ///
	, legend(off) xtitle("x") ytitle("Prob(#Employees{&ge}x)") title("Treatment Effect on Prob(#Employees{&ge}x)")
graph export "${pset_folder}/distribution_effects.png", replace
graph export "${pset_folder}/distribution_effects.eps", replace

capture drop coeff std_error x upper_ci lower_ci
