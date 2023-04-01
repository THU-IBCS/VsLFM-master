% 3D reconstruciton for VsLFM
% The Code is created based on the method described in the following paper 
%        ZHI LU, YU LIU etc.
%        Virtual-scanning light-field microscopy for robust snapshot high-resolution volumetric imaging
%        Nature Methods, 2023.
% 
%    Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
%    Date  : 06/01/2022

clear;

addpath('./Solver/');
addpath('./utils/');

% Preparameters
GPUcompute=1;        %% GPU accelerator (on/off)
Nnum=13;             %% the number of sensor pixels after each microlens/ the number of angles in one dimension
DAO = 0;             %% digital adaptive optics (on/off)
Nb=1;                %% number of blocks for multi-site AO in one dimension
VsNet_Enabled = 1;   %% Vs-Net for resolution enhancement (on/off)

% PSF
load('PSF/Experimental_psf_M63_NA1.4_zmin-8u_zmax8u.mat','psf');
weight=squeeze(sum(sum(sum(psf,1),2),5))./sum(psf(:));
weight=weight-min(weight(:));
weight=weight./max(weight(:)).*0.8;
for u=1:Nnum
    for v=1:Nnum
        if (u-round(Nnum/2))^2+(v-round(Nnum/2))^2>(round(Nnum/3))^2 
            weight(u,v)=0;
        end
    end
end

% Load spatial angular components
if VsNet_Enabled == 1
    Nshift=3;  % the sampling points of a single scanning period
    maxIter=10;% the maximum iteration number of single frame
    WDF=zeros(363,363,Nnum,Nnum);
    for u=1:Nnum
        for v=1:Nnum
            tmp=single(imread(['Data/SR/mito_SR.tif'],(u-1)*Nnum+v));
            WDF(:,:,u,v)=tmp(97:459,70:432); %% content-aware FOV
        end
    end
else
    Nshift=1;  % the sampling points of a single scanning period
    maxIter=1; % the maximum iteration number of single frame
    WDF=zeros(121,121,Nnum,Nnum);
    for u=1:Nnum
        for v=1:Nnum
            tmp=single(imread(['Data/LR/mito_LR.tif'],(u-1)*Nnum+v));
            WDF(:,:,u,v)=tmp(33:153,24:144); %% content-aware FOV
        end
    end
end
    
% Initialization
WDF=imresize(WDF,[size(WDF,1)*Nnum/Nshift,size(WDF,2)*Nnum/Nshift]);
Xguess=ones(size(WDF,1),size(WDF,2),size(psf,5));
Xguess=Xguess./sum(Xguess(:)).*sum(WDF(:))./(size(WDF,3)*size(WDF,4));

% 3D Reconstruction
tic;
Xguess = deconvRL(maxIter, Xguess,WDF, psf, weight, DAO, Nb, GPUcompute);
ttime = toc;

% Save high-resolution reconstructed volume
mkdir('Data/Reconstruction');
imwriteTFSK(single(gather(Xguess(545:1295,20:1260,21:51))),['Data/Reconstruction/mito_Recon.tif']);  %% crop volume edge and save it



