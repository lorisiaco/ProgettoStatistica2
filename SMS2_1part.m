T= readtable("GoldUp.csv");
%--------------------------SALVATAGGIO VARIABILI-------------------------
%scelgo la Y(Gold_Price)
Y=T{:,2}
%carico nei vettori le variabili per effettuare i vari confronti
X1=T{:,3}      %Crude_Oil
X2=T{:,4}      %Interest_Rate
X3=T{:,5}      %USD_INR
X4=T{:,6}      %Sensex
X5=T{:,7}      %CPI
X6=T{:,8}      %USD_Index
X_Tot=[[X1],[X2],[X3],[X4],[X5],[X6]]
%--------------------------RICERCA MIGLIORE MODELLO--------------------------------
%% Domanda1: ANALIZZO COME IL PREZZO DELL'ORO SI COMPORTA CON GLI ALTRI REGRESSORI
A=fitlm(X1,Y)       %Crude_Oil con Gold_Price
%grafico
plot(X1, Y, 'o')
title('Crude_Oil')
xlabel('X1 ($)')
ylabel('Y ($)')
B=fitlm(X2,Y)       %Interest_Rate con Gold_Price
%grafico
plot(X2, Y, 'o')
title('Interest_Rate')
xlabel('X2 ($)')
ylabel('Y ($)')
C=fitlm(X3,Y)       %USD_INR con Gold_Price
%grafico
plot(X3, Y, 'o')
title('USD_INR')
xlabel('X3 ($)')
ylabel('Y ($)')
D=fitlm(X4,Y)       %Sensex con Gold_Price
%grafico
plot(X4, Y, 'o')
title('Sensex')
xlabel('X4 ($)')
ylabel('Y ($)')
E=fitlm(X5,Y)       %CPI con Gold_Price
%grafico
plot(X5, Y, 'o')
title('CPI')
xlabel('X5 ($)')
ylabel('Y ($)')
F=fitlm(X6,Y)       %USD_Index con Gold_Price
%grafico
plot(X5, Y, 'o')
title('USD_Index')
xlabel('X5 ($)')
ylabel('Y ($)')
%----------------MODELLO REGRESSIONE--------------------------------------
Xm1=[[X5],[X4],[X3]]        %costruisco una matrice con i regressori con R2 piu alto
MOD1=fitlm(Xm1,Y)       %R2 del 0.95 
Xm2=[[X5],[X4]]
MOD2=fitlm(Xm2,Y)       %R2 del 0.92
Xm3=[[X5],[X3]]
MOD3=fitlm(Xm3,Y)       %R2 del 0.929
Xm4=[[X3],[X4]]
MOD4=fitlm(Xm4,Y)       %R2 del 0.84
%Modello lineare scelto é il MOD1 con regressori CPI,Sensex,USD_INR
plot(MOD1)
%-----------------------------ANALISI MODELLO(MOD1)------------------------
R2=MOD1.Rsquared.Adjusted           %Vale n0.9529 (r2 aggiustato)
PV=MOD1.Coefficients.pValue         %i p-value sono bassi e rifiutano H0 tranne il p-value dell'intercetta
Ftes=coefTest(MOD1)                 %Molto piccolo va bene
Beta_cap=MOD1.Coefficients.Estimate
%% Domanda 2: Esiste multicollinearitá?
%--------------VERIFICA CONDIZIONE DI ESISTENZA---------------------------
X_varGold=[[X3]]
uno=ones(239,1)
Z=[uno,X_varGold]
Z1=Z'
determinanteGold=det(Z1*Z)      %positivo/ va bene, non c'é multicollinearitá
%% Domanda 3: I residui sono normali ed hanno media pari a 0?
%--------------VERIFICA DELLA NORMALITA' DEGLI ERRORI CON TEST JB-----
Residui = MOD1.Residuals.Raw
Media=mean(Residui) %Media dei residui bassa tendente a 0
plot(Residui)
title('Residui')
yline(0,'r','LineWidth',3)
yline(Media,'b','LineWidth',2)
%JB test
h = jbtest(Residui)
histfit(Residui)
xlabel('CPI,Sensex,USD_INR')
ylabel('GoldPrice')
%h é 1 quindi gli errori hanno distribuzione normale
%-------------------------------------------------------------------------
%% Domanda4:Esiste una relazione tra Crude_Oil e USD_Index?
plot(X6,X1,'o')     %Osserviamo come sono distrubiti i dati
ordine = 6;
nbasis = length(X6) + ordine - 2        %numero basi=ordine+numero di knots interiori
rangevalue = [min(X6) max(X6)];         %e il  nkontsinteriori= lunghezza(knts)-2
USDBasis = create_bspline_basis(rangevalue, nbasis, ordine);
nderiv = 2; % derivata seconda
lambda = 0.01;
basismat1 = eval_basis(X6, USDBasis);   

Rmat = eval_penalty(USDBasis, nderiv);
USDfdpar = fdPar(USDBasis, nderiv, lambda, 1, Rmat);
[fdobj, df, gcv, coef, SSE, penmat, y2cmap] = smooth_basis(X6, X1, USDfdpar);
chat_s = y2cmap*X1;              %calcolo di chat(la stima del vettore dei coefficienti)
Smat_s = basismat1 * y2cmap;     %calcolo della matrice di smoothing come nelle slide
yhat_s = Smat_s * X1;            %stima di y cappello con spline

nobs = length(X6);
df = trace(Smat_s);     %gradi di libertà 
RSS1 = sum((X1 - yhat_s).^2);       %Errore quadratico medio
%Stima della Varianza
sigma_hat_s = RSS1 / (nobs - df);
stdDevhat_s = sqrt(sigma_hat_s);

loglam = -6:0.25:0;     %fornisco un vettore di valore plausibili per i logaritmi dei lambda
gcvsave = zeros(length(loglam), 1);     %vettore per salvare GCV
dfsave = gcvsave;           %creo un vettore per prendere il grado di libertà efettivo in funzione di lambda
lambdai = 0;
%smoothing spline con valore di lambda calcolato tramite cross-validazione
for i = 1:length(loglam)
    lambdai = 10^loglam(i);     %prendo lambda 
    USDfdPari = fdPar(USDBasis, 4, lambdai);
    [fdobji, dfi, gcvi] = smooth_basis(X6, X1, USDfdPari);
    gcvsave(i) = sum(gcvi);
    dfsave(i) = dfi;
end
plot(loglam, gcvsave, '-o');
ylabel('sum(GCV)')
xlabel('lambda')
apicel = find(gcvsave == min(gcvsave));    %lambda che minimizza la funzione lambda^indicel(25)
%Intervallo di confidenza a 95%
var_cov_yhat = sigma_hat_s*(Smat_s'*Smat_s);     %matrice var-cov dei valori stimati
varYhat = diag(var_cov_yhat);   %Prendiamo le varianze contenute sulla diagonale della matrice
alpha = 0.05;
z_alpha = norminv(1 - (alpha)/2);
Lower = yhat_s - z_alpha*sqrt(varYhat);
Upper = yhat_s + z_alpha*sqrt(varYhat);
figure
plot( Lower, '--k', 'LineWidth', 2)
hold on
plot( Upper, '--k', 'LineWidth', 2)
hold on
plotfit_fd(X1 , X6, fdobj)      %Funzione stimata con smooth e CV
ylabel('CrudeOil')
xlabel('USDIndex')
legend( 'LowerIC limite', 'UpperIC limite' ,'valori reali','funzione stimata con CV')
hold off
%-------------------------------------------------------------------------
%% Domanda5:E' possibile esprimere il modello scelto al primo punto tramite WLS?
%Gold_Price=Y
%CPI,Sensex,USD_INR=Xm1
plot(Xm1,Y)         %Osserviamo come sono distrubiti i dati
figure
MIN= min(Xm1);     %Massimo 
MAX= max(Xm1);     %Minimo
XW= MIN:0.01:MAX;                                                                               
X11=[ones(length(Xm1),1) Xm1];
BH = (X11'*X11)^(-1)*X11'*Y;
YW = XW*BH(2)+BH(1);
O=ones(length(X11),1);       %Suppongo che i dati che ho siano corretti
plot(XW,YW)
hold on
LBH=[0 0]';         %Pongo tutto uguale a 0
a=0;       %Mi serve per verificare le condizioni
c=1;        %Contatore
while a==0          %Pesi scelti con algoritmo iterativo
    L(c,:) = O;
    O=diag(O);
    BH=inv(X11'*inv(O)*X11)*X11'*inv(O)*Y; 
    Residui= Y-(BH(1)+BH(2)*X11);         %residui tramite formula slide
    O=abs(Residui)'+0.001;      %valore assoluto + quantita' minuscola
    YW=XW*BH(2)+BH(1);      %formula slide
    plot(XW,YW)
    YW1=YW;
    d= norm(L-BH);
    if d<0.001
        a=1;
    end
    c=c+1;      %Incremento il contatore
    L=BH;       %Ho calcolato la distanza
end