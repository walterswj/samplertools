# -*- coding: utf-8 -*-
"""
Some tools for manipulating Sampler output.

Created on Thu Feb 24 12:29:31 2022.

@author: William Walters
"""

import numpy as np
import glob
import h5py

class SamplerDatabase():
    """Class for holding a database of sampler calculations."""
        
    def __init__(self, h5_name=None):
        if (h5_name==None):
            self.cases=np.array([])
            self.t=np.array([])
            self.nucs=np.array([])
            self.data=np.array([])
            self.nsamples=0
            self.nt=0
            self.ncases=0
            self.nnucs=0
            self.mean=np.array([])
            self.unc=np.array([])
            self.runc=np.array([])
        else:
            with h5py.File(h5_name,'r') as f:
                self.cases=f['cases'][()]
                self.t=f['t'][()]
                self.nucs=f['nucs'][()]
                self.data=f['data'][()]
                self.nsamples=self.data.shape[0]
                self.nnucs=self.data.shape[1]
                self.ncases=self.data.shape[2]
                self.nt=self.data.shape[3]
                #self.mean=f['mean'][()]
                #self.unc=f['unc'][()]
                #self.runc=f['runc'][()]
                self.calculate_unc()
                
    def read_f71_csv(self,fname):
        """Read a single f71-csv file and adds it to the database.

        Parameters
        ----------
        fname : str
            Path to the csv file.

        Returns
        -------
        dats : ndarry
            All values, shape is (num_cases*len(t),len(nucs)).
        nucs : ndarray
            List of nuclides, formatting depends on how the csv was generated.
        t : ndarray
            Unique timesteps in the problem.
        cases : ndarray
            The cases array, should be length number of cases*len(t).

        """   
        print(fname)
        dat=np.genfromtxt(fname,delimiter=',',autostrip=True)
        cases=dat[0,:] # case number is the first row
        case_inds = cases==cases[1] #this is the indices of the first case (to extract the unique timesteps)
        t=dat[2,case_inds] #timesteps are in the 3rd row, and are repeated for each case (only take 1st case)
        nt=len(t)
        cases=cases[np.arange(1,len(cases),nt)] # all cases, past first column
        ncases=len(cases)
        nucs=dat[6:,0] # actual data starts after 6 header rows, first column is nuclide ID
        nnucs=len(nucs)
        dats=dat[6:,1:] # nuclide conc. is everything past first column
        dats.shape=(1,nnucs,ncases,nt)
        self.nsamples+=1 # keep track of count the number of samplers
        
        if (len(self.data)==0):
            self.data=dats
            
            #these should only be stored once, as they are the same between all cases
            self.nucs=nucs
            self.t=t
            self.cases=cases
        else:
            self.data=np.append(self.data,dats,axis=0)
        return dats,nucs,t,cases

    def calculate_unc(self):
        """Calculate the mean and uncertainty for each nuclide.
        
        This creates the variables unc, mean, and runc.

        Returns
        -------
        None.

        """
        self.unc=np.std(self.data[1:,...],axis=0,ddof=1)
        self.mean=np.mean(self.data[1:,...],axis=0)
        
        #ignore warnings from 0/0 - this happens for many nuclides
        with np.errstate(divide='ignore',invalid='ignore'):
            self.runc=self.unc/self.mean
        
    def get_data(self,nuc,case):
        """Get the nuclide concentration data for a given case and nuclide.

        Parameters
        ----------
        nuc : int or ndarray
            Integer array of nuclide IDs, or single ID.
        case : int or ndarray
            Integer array of cases, or single case.

        Returns
        -------
        ndarray
            Array of nuclide cconcentrations with shape (len(nuc),len(case)).

        """
        inucs=np.in1d(self.nucs,nuc)
        icases=np.in1d(self.cases,case)
        return np.squeeze(self.data[1:,inucs,icases,:])
    
    #get the mean of a given nuclide for a given case
    #does not work with lists of nuclides or cases
    def get_mean(self,nuc,case):
        """Get mean values over time for a given nuclide and case."""
        return np.squeeze(self.mean[self.nucs==nuc,self.cases==case,:])
    
    #get the uncertainty (*not relative*) for a given nuclide and case
    def get_unc(self,nuc,case):
        """Get uncertainty over time for a given nuclide and case."""
        return np.squeeze(self.unc[self.nucs==nuc,self.cases==case,:])
    
    def gen_h5_from_csv(self,pathname,outfname):
        """
        Generate an hdf5 database file from a set of f71 csv files.

        Parameters
        ----------
        pathname : str
            Path expression including * that includes all csv files.
            glob.glob is used to extract all relevant files from pathname
            e.g., pathname='some_model/test.samplerfiles/test*.f71.csv'
            csv files can be generated from f71 using f71tocsv or obiwan
        outfname : str
            Output h5 filename.

        Returns
        -------
        None.

        """
        newDB=SamplerDatabase()
        files=sorted(glob.glob(pathname))
        for f in files:
            newDB.read_f71_csv(f)
        newDB.calculate_unc()
        newDB.save_to_h5(outfname)
    
    def save_to_h5(self,fname):
        """
        Save the current database to an hdf5 file.

        Parameters
        ----------
        fname : str
            Output file name.

        Returns
        -------
        None.

        """
        with h5py.File(fname,'w') as f:
            f.create_dataset('data',data=self.data,dtype='float32')
            f.create_dataset('nucs',data=self.nucs,dtype='int32')
            f.create_dataset('cases',data=self.cases,dtype='int32')
            f.create_dataset('t',data=self.t,dtype='float32')
            f.create_dataset('nsamples',data=self.nsamples,dtype='int32')
            f.create_dataset('nt',data=self.nt,dtype='int32')
            f.create_dataset('ncases',data=self.ncases,dtype='int32')
            f.create_dataset('nnucs',data=self.nnucs,dtype='int32')
            #f.create_dataset('mean',data=self.mean,dtype='float32')
            #f.create_dataset('unc',data=self.unc,dtype='float32')
            #f.create_dataset('runc',data=self.runc,dtype='float32')
    
    def select_nuclides(self,nuc_indices):
        """
        Create a new database with reduced set of nuclides.

        Parameters
        ----------
        nuc_indices : bool array
            Array of length equal to nucs indicating which nuclides to keep.

        Returns
        -------
        newDB : SamplerDatabase
            A new database containing the reduced dataset.
        
        Example
        -------
        sdb=SamplerDatabase('db.h5')
        inds=sdb.mean[:,0,-1]>1e-15 #select nuclides above a cutoff
        sdb_reduced=sdb.select_nuclides(inds)
        sdb_reduced.save_to_h5('db_reduced.h5')
        

        """
        newDB=self
        newDB.nucs=newDB.nucs[nuc_indices]
        newDB.nnucs=len(newDB.nucs)
        newDB.data=newDB.data[:,nuc_indices,:,:]
        newDB.calculate_unc()
        
        return newDB

def gen_h5_from_csv(pathname,outfname):
    """Generate an HDF5 database from a set of sampler f71 csv files.
    
    Parameters
    ----------
    pathname : str
        Pathname string, with wildcards '*' that includes all csv files.
        This uses glob.glob(pathname) to get the list of file names.
        Ex. pathname='somde_model/test.samplerfiles/test*.f71.csv'
        csv files can be generated from f71 using f71tocsv or obiwan
    outfname : str
        Output file name, should end in '.h5'.

    Returns
    -------
    None.
    
    Notes
    -----
    This creates a file with the name outfname

    """
    newDB=SamplerDatabase()
    files=sorted(glob.glob(pathname))
    for f in files:
        newDB.read_f71_csv(f)
    newDB.calculate_unc()
    newDB.save_to_h5(outfname)    

