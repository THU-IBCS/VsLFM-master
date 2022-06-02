function [index1,index2] = angle_order(Nnum)
% Auto angle order generation for tomographic reconstrcution
%
% The Code is created based on the method described in the following paper 
%   [1]  ZHI LU etc,
%        "A practical guide to scanning light-field microscopy with digital adaptive optics"
%        accepted by Nature Protocols, 2022
%
%    Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
%    Date  : 10/01/2021

[xx,yy]=meshgrid(1:Nnum+2, 1:Nnum+2);
index1=[];
index2=[];
for i=1:round((1+Nnum)/2)
    endXY=Nnum+1-i;
    for j=i:endXY
        index1=[index1,yy(i,j)];
        index2=[index2,xx(i,j)];
    end
    for j=i+1:endXY
        index1=[index1,yy(j,endXY)];
        index2=[index2,xx(j,endXY)];
    end
    if i<endXY
        for j=endXY-1:-1:i
            index1=[index1,yy(endXY,j)];
            index2=[index2,xx(endXY,j)];
        end
    end
    if i<endXY
        for j=endXY-1:-1:i+1
            index1=[index1,yy(j,i)];
            index2=[index2,xx(j,i)];
        end
    end
end

end
