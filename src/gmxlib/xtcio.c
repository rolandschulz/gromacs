/*
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * GROningen Mixture of Alchemy and Childrens' Stories
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "typedefs.h"
#include "xdrf.h"
#include "gmxfio.h"
#include "xtcio.h"
#include "smalloc.h"
#include "vec.h"
#include "futil.h"
#include "gmx_fatal.h"

#include <domdec.h>
#include <mpi.h>  //TODO: make wrapper to have MPI functions not in this file
#include <rpc/xdr.h>  //TODO: make wrapper to have XDR functions not in this file

#define XTC_MAGIC 1995


static int xdr_r2f(XDR *xdrs,real *r,gmx_bool bRead)
{
#ifdef GMX_DOUBLE
    float f;
    int   ret;
    
    if (!bRead)
      f = *r;
    ret = xdr_float(xdrs,&f);
    if (bRead)
      *r = f;
    
    return ret;
#else
    return xdr_float(xdrs,(float *)r);
#endif
}

MPI_File open_xtc(const char *fn,const char *mode, gmx_domdec_t *dd)
{
	MPI_File fh = NULL;
	MPI_Comm new_comm;
	int amode;

	if (strcmp(mode,"w+")==0) {
		amode = MPI_MODE_RDWR | MPI_MODE_CREATE;
	} else if (strcmp(mode,"a+")==0) {
		amode = MPI_MODE_RDWR | MPI_MODE_APPEND;
	} else if (strcmp(mode,"r")==0) {
		amode = MPI_MODE_RDONLY;
	} else {
		gmx_fatal(FARGS,"Unknown mode!");
	}
    if (mode[0]=='w')
    {
        /* only make backups for normal gromacs */
        make_backup(fn);
    }
	MPI_Comm_split(dd->mpi_comm_all, dd->rank < NUMBEROFSTEPS, dd->rank, &new_comm );// new_comm must be a vector of size color
	if (dd->rank < NUMBEROFSTEPS)
	{
		if (MPI_File_open(new_comm,(char*)fn,amode,MPI_INFO_NULL, &fh) != MPI_SUCCESS)
		{
			return NULL;
		}
	}
	return fh;
    //return gmx_fio_open(fn,mode);
}

void close_xtc(MPI_File fio)
{
	if (fio!=NULL) {
		MPI_File_close(&fio);
	}
    //gmx_fio_close(fio);
}

static void check_xtc_magic(int magic)
{
  if (magic != XTC_MAGIC) 
    gmx_fatal(FARGS,"Magic Number Error in XTC file (read %d, should be %d)",
		magic,XTC_MAGIC);
}

int xtc_check(const char *str,gmx_bool bResult,const char *file,int line)
{
  if (!bResult) {
    if (debug)
      fprintf(debug,"\nXTC error: read/write of %s failed, "
	      "source file %s, line %d\n",str,file,line);
    return 0;
  }
  return 1;
}

void xtc_check_fat_err(const char *str,gmx_bool bResult,const char *file,int line)
{
  if (!bResult) {
    gmx_fatal(FARGS,"XTC read/write of %s failed, "
		"source file %s, line %d\n",str,file,line);
  }
}

static int xtc_header(XDR *xd,int *magic,int *natoms,int *step,real *time,
		      gmx_bool bRead,gmx_bool *bOK)
{
  int result;

  if (xdr_int(xd,magic) == 0)
    return 0;
  result=XTC_CHECK("natoms", xdr_int(xd,natoms));  /* number of atoms */
  if (result)
    result=XTC_CHECK("step",   xdr_int(xd,step));    /* frame number    */
  if (result)
    result=XTC_CHECK("time",   xdr_r2f(xd,time,bRead));   /* time */
  *bOK=(result!=0);

  return result;
}

static int xtc_coord(XDR *xd,int *natoms,matrix box,rvec *x,real *prec, gmx_bool bRead)
{
  int i,j,result;
#ifdef GMX_DOUBLE
  float *ftmp;
  float fprec;
#endif
    
  /* box */
  result=1;
  for(i=0; ((i<DIM) && result); i++)
    for(j=0; ((j<DIM) && result); j++)
      result=XTC_CHECK("box",xdr_r2f(xd,&(box[i][j]),bRead));

  if (!result)
      return result;
  
#ifdef GMX_DOUBLE
  /* allocate temp. single-precision array */
  snew(ftmp,(*natoms)*DIM);
  
  /* Copy data to temp. array if writing */
  if(!bRead)
  {
      for(i=0; (i<*natoms); i++)
      {
          ftmp[DIM*i+XX]=x[i][XX];      
          ftmp[DIM*i+YY]=x[i][YY];      
          ftmp[DIM*i+ZZ]=x[i][ZZ];      
      }
      fprec = *prec;
  }
  result=XTC_CHECK("x",xdr3dfcoord(xd,ftmp,natoms,&fprec));
  
  /* Copy from temp. array if reading */
  if(bRead)
  {
      for(i=0; (i<*natoms); i++)
      {
          x[i][XX] = ftmp[DIM*i+XX];      
          x[i][YY] = ftmp[DIM*i+YY];      
          x[i][ZZ] = ftmp[DIM*i+ZZ];      
      }
      *prec = fprec;
  }  
  sfree(ftmp);
#else
    result=XTC_CHECK("x",xdr3dfcoord(xd,x[0],natoms,prec)); 
#endif 
    
  return result;
}



int write_xtc(MPI_File fio,
	      int natoms,int step,real time,
	      matrix box,rvec *x,real prec, gmx_bool bDontWrite)
{
  int magic_number = XTC_MAGIC;
  static char *mem_buf = NULL;
  static XDR *xd;
  gmx_bool bDum;
  u_int nBytes;
  MPI_Status status;
  int bOK;
	
  if (mem_buf == NULL)
  {
	  snew (mem_buf, 3* natoms * sizeof(real));
	  snew(xd, 1);
	  xdrmem_create(xd,mem_buf,3* natoms * sizeof(real),XDR_ENCODE);
  }

  if (bDontWrite) {
	  bOK = MPI_File_write_ordered(fio,mem_buf,0,MPI_BYTE,&status) == MPI_SUCCESS;
	  return bOK;
  }

  xdr_setpos(xd,0);


  //xd = gmx_fio_getxdr(fio);
  /* write magic number and xtc identidier */
  if (xtc_header(xd,&magic_number,&natoms,&step,&time,FALSE,&bDum) == 0)
  {
	  return 0;
  }
    
  /* write data */
  bOK = xtc_coord(xd,&natoms,box,x,&prec,FALSE); /* bOK will be 1 if writing went well */

//  if(bOK)
//  {
//	  if(gmx_fio_flush(fio) !=0)
//	  {
//		  bOK = 0;
//	  }
//  }

  nBytes = xdr_getpos(xd);
  if (bOK)
  {
	  bOK = MPI_File_write_ordered(fio,mem_buf,nBytes,MPI_BYTE,&status) == MPI_SUCCESS;
  }

  return bOK;  /* 0 if bad, 1 if writing went well */
}

int read_first_xtc(t_fileio *fio,int *natoms,int *step,real *time,
		   matrix box,rvec **x,real *prec,gmx_bool *bOK)
{
  int magic;
  XDR *xd;
  
  *bOK=TRUE;
  xd = gmx_fio_getxdr(fio);
  
  /* read header and malloc x */
  if ( !xtc_header(xd,&magic,natoms,step,time,TRUE,bOK))
    return 0;
    
  /* Check magic number */
  check_xtc_magic(magic);
  
  snew(*x,*natoms);

  *bOK=xtc_coord(xd,natoms,box,*x,prec,TRUE);
  
  return *bOK;
}

int read_next_xtc(t_fileio* fio,
		  int natoms,int *step,real *time,
		  matrix box,rvec *x,real *prec,gmx_bool *bOK)
{
  int magic;
  int n;
  XDR *xd;

  *bOK=TRUE;
  xd = gmx_fio_getxdr(fio);
  
  /* read header */
  if (!xtc_header(xd,&magic,&n,step,time,TRUE,bOK))
    return 0;

  /* Check magic number */
  check_xtc_magic(magic);

  if (n > natoms) {
    gmx_fatal(FARGS, "Frame contains more atoms (%d) than expected (%d)", 
	      n, natoms);
  }

  *bOK=xtc_coord(xd,&natoms,box,x,prec,TRUE);

  return *bOK;
}


