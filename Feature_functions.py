from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.io.vasp import Vasprun
from pymatgen.ext.matproj import MPRester
import pandas as pd
import os
import numpy as np
import warnings

################# Functions to calculate structure features
# These functions take in a structure object and calculate various structure features

def dimensionality(structure):
    from matminer.featurizers.structure import Dimensionality
    
    # Initialize the featurizer
    sc = Dimensionality()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)


def symmetry(structure):
    from matminer.featurizers.structure import GlobalSymmetryFeatures

    # Initialize the featurizer
    sc = GlobalSymmetryFeatures()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def sites(structure):
    from matminer.featurizers.structure import SiteStatsFingerprint

    # Initialize the featurizer
    sc = SiteStatsFingerprint()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def BranchPointEnergy(structure):
    from matminer.featurizers.bandstructure import BranchPointEnergy

    # Initialize the featurizer
    sc = BranchPointEnergy()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

#I don't know if this is relevant since it gives the same information as earlier functions
def multiple_structure(structure):
    from matminer.featurizers.structure import GlobalSymmetryFeatures, Dimensionality
    from matminer.featurizers.base import MultipleFeaturizer

    # Create a MultipleFeaturizer with some featutrizers
    sc = MultipleFeaturizer([
        GlobalSymmetryFeatures(),
        Dimensionality()
    ])

    # Featurize
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()

    return features, feature_labels

def bond_fractions(structure):
    from matminer.featurizers.structure.bonding import BondFractions

    # Initialize the featurizer
    sc = BondFractions()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()

    return features, feature_labels

def bag_of_bonds(structure):
    from matminer.featurizers.structure.bonding import BagofBonds

    # Initialize the featurizer 
    sc = BagofBonds()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])  

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def global_instability_index(vasprun):
    from matminer.featurizers.structure.bonding import GlobalInstabilityIndex
    from pymatgen.analysis.bond_valence import BVAnalyzer

    #This function needed vasprun to work
    structure = vasprun.final_structure

    # Assign oxidation states
    structure_with_oxi = BVAnalyzer().get_oxi_state_decorated_structure(structure)


    # Featurize
    sc = GlobalInstabilityIndex()
    features = sc.featurize(structure_with_oxi)
    feature_labels = sc.feature_labels()
    return features, feature_labels

#This feature does not work on 2D because of VoronoiNN
def structural_heterogeneity(structure):
    from matminer.featurizers.structure import StructuralHeterogeneity

    try:
        sc = StructuralHeterogeneity()
        features = sc.featurize(structure)
        labels = sc.feature_labels()
        return features, labels
    except Exception as e:
        print("StructuralHeterogeneity skipped – likely incompatible with 2D structure.")
        print(f"Reason: {e}")
        return [], []


def minimum_relative_distances(structure):
    from matminer.featurizers.structure import MinimumRelativeDistances

    sc = MinimumRelativeDistances()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])

    features = sc.featurize(structure)
    labels = sc.feature_labels()
    return features, labels

def jarvis_cfid(structure):
    from matminer.featurizers.structure.composite import JarvisCFID
    
    cs = JarvisCFID()
    
    features = cs.featurize(structure)
    labels = cs.feature_labels()
    return (features, labels)
    

def coulomb_matrix(structure):
    from matminer.featurizers.structure.matrix import CoulombMatrix

    sc = CoulombMatrix()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])
    
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def sine_coulomb_matrix(structure):
    from matminer.featurizers.structure.matrix import SineCoulombMatrix

    sc = SineCoulombMatrix()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])

    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def orbital_field_matrix(structure):
    from matminer.featurizers.structure.matrix import OrbitalFieldMatrix

    sc = OrbitalFieldMatrix()
    # fit is used to initialize internal parameters of the featurizer based on the input structure
    sc.fit([structure])

    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def ewald_energy(vasprun):
    from matminer.featurizers.structure.misc import EwaldEnergy
    from pymatgen.analysis.bond_valence import BVAnalyzer

    #This function did not work without vasprun
    structure = vasprun.final_structure
    #Assign oxidation states
    structure_with_oxi = BVAnalyzer().get_oxi_state_decorated_structure(structure)

    sc = EwaldEnergy()
    
    features = sc.featurize(structure_with_oxi)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def structure_composition(structure):
    from matminer.featurizers.structure.misc import StructureComposition
    from matminer.featurizers.composition import ElementProperty

    # Generate composition-based features using the Magpie preset.
    composition_feat = ElementProperty.from_preset("magpie")

    sc = StructureComposition(composition_feat)

    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()

    return features, feature_labels

def xrd_powder_pattern(structure):
    from matminer.featurizers.structure import XRDPowderPattern

    sc = XRDPowderPattern()

    sc.fit([structure])  

    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()

    return features, feature_labels

############ Site Features
##### These functions take in structure and idx's of composition to get site features 

def BondOrientationalParameter(structure, idx):
    from matminer.featurizers.site import BondOrientationalParameter

    # Initialize the featurizer
    sc = BondOrientationalParameter()

    # Featurize the structure
    features = sc.featurize(structure, idx)
    feature_labels = sc.feature_labels()
    return (features, feature_labels) 

def AverageBondLength(structure, idx):
    from matminer.featurizers.site import AverageBondLength
    from pymatgen.analysis.local_env import CrystalNN #Length measuring requires a method
    #method = CrystalNN() #(arbitrarily) chosen method

    from pymatgen.analysis.local_env import VoronoiNN
    method = VoronoiNN(allow_pathological=True)


    # Initialize the featurizer
    sc = AverageBondLength(method=method)

    try:
        features = sc.featurize(structure, idx)
        feature_labels = sc.feature_labels()
    except IndexError as e:
        #print(f"Site {idx} in {variant} skipped: {e}")
        feature_labels = sc.feature_labels()
        features = 'Error'
    return (features, feature_labels) 

def AverageBondAngle(structure, idx):
    from matminer.featurizers.site import AverageBondAngle
    from pymatgen.analysis.local_env import CrystalNN #Length measuring requires a method
    method = CrystalNN() #(arbitrarily) chosen method
    
    # Initialize the featurizer
    sc = AverageBondAngle(method = method)

    try:
        features = sc.featurize(structure, idx)
        feature_labels = sc.feature_labels()
    except IndexError as e:
        #print(f"Site {idx} in {variant} skipped: {e}")
        feature_labels = sc.feature_labels()
        features = 'Error'
    return (features, feature_labels) 


def ChemicalSRO(structure, idx):
    from matminer.featurizers.site import ChemicalSRO
    from pymatgen.analysis.local_env import VoronoiNN  # Import VoronoiNN for the nearest neighbor method
    
    NN = VoronoiNN() # Nearest neighbor method
    # Initialize the featurizer
    sc = ChemicalSRO(nn = NN)

    # Fits the data to the structure
    sc.fit([(structure, idx)])
        
    # Featurize the structure
    features = sc.featurize(structure, idx)
    feature_labels = sc.feature_labels()
    return (features, feature_labels) 


def EwaldSiteEnergy(structure, idx):
    from matminer.featurizers.site import EwaldSiteEnergy
    
    
    # Initialize the featurizer
    sc = EwaldSiteEnergy()

    # Featurize the structure
    features = sc.featurize(structure, idx)
    feature_labels = sc.feature_labels()
    return (features, feature_labels) 


def LocalPropertyDifference(structure, idx):
   from matminer.featurizers.site import LocalPropertyDifference
   # from pymatgen.analysis.local_env import CrystalNN
   # NN = CrystalNN()

  # Initialize the featurizer
   sc = LocalPropertyDifference()

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)


def SiteElementalProperty(structure, idx):
   from matminer.featurizers.site import SiteElementalProperty

  # Initialize the featurizer
   sc = SiteElementalProperty()

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)


def SOAP(structure, idx):
   from matminer.featurizers.site import SOAP as soap

   # Initialize the featurizer
   sc =  soap(rcut = 5.0, nmax = 8, lmax = 6, sigma=0.5, periodic=True)
   sc.fit([structure])
   
   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)


def AGNIFingerprints(structure, idx):
   from matminer.featurizers.site import AGNIFingerprints

  # Initialize the featurizer
   sc = AGNIFingerprints()

   # Featurize the structure
   features = sc.featurize(structure, idx )
   feature_labels = sc.feature_labels()

   return (features, feature_labels)

def OPSiteFingerprint(structure, idx):
   from matminer.featurizers.site import OPSiteFingerprint

  # Initialize the featurizer
   sc = OPSiteFingerprint()

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)

def CrystalNNFingerprint(structure, idx):
   from matminer.featurizers.site import CrystalNNFingerprint

   #op_types = ['coordination_number', 'ang_dist']    
   op_types = {
        3: ['wt', 'cn'],  #
        4: ['wt', 'cn'],  #
   }
   
   # Initialize the featurizer
   sc = CrystalNNFingerprint( op_types = op_types  )
   

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)


def VoronoiFingerprint(structure, idx):
   from matminer.featurizers.site import VoronoiFingerprint

  # Initialize the featurizer
   sc = VoronoiFingerprint()

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)




def chem_env_features(structure, idx):  
    from matminer.featurizers.site import ChemEnvSiteFingerprint

    # Initialize the featurizer
    sc = ChemEnvSiteFingerprint.from_preset("simple")  # or "multi_weights"
    
    # Featurize the structure
    features = sc.featurize(structure, idx)
    feature_labels = sc.feature_labels()
    return features, feature_labels

def IntersticeDistribution(structure, idx):  
    from matminer.featurizers.site import IntersticeDistribution

    # Initialize the featurizer
    sc = IntersticeDistribution() # or "multi_weights"
    
    # Featurize the structure
    feature_labels = sc.feature_labels()
    try:
        features = sc.featurize(structure, idx)
    except ValueError as e:
        #print(f"Site {idx} in {variant} skipped: {e}")
        features = 'Error'
    return features, feature_labels

def CoordinationNumber(structure, idx):
   from matminer.featurizers.site import CoordinationNumber

  # Initialize the featurizer
   sc = CoordinationNumber()

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)

def GaussianSymmFunc(structure, idx):
   from matminer.featurizers.site import GaussianSymmFunc

  # Initialize the featurizer
   sc = GaussianSymmFunc()

   # Featurize the structurea
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)

def GeneralizedRadialDistributionFunction(structure, idx):
   from matminer.featurizers.site import GeneralizedRadialDistributionFunction
   from matminer.featurizers.site.rdf import Gaussian

   # Define Gaussian functions as bins
   bins = [Gaussian(center, 0.5) for center in np.linspace(1, 4.5, 10)]

    
   # Initialize the featurizer
   sc = GeneralizedRadialDistributionFunction(bins = bins)  # resolution 

   # Featurize the structure
   features = sc.featurize(structure, idx)
   feature_labels = sc.feature_labels()

   return (features, feature_labels)


def AngularFourierSeries(structure, idx):
    from matminer.featurizers.site import AngularFourierSeries
    from matminer.featurizers.site.rdf import Gaussian

    # Define Gaussian functions as bins
    bins = [Gaussian(center, 0.5) for center in np.linspace(1, 4.5, 10)]

    # Initialize the featurizer
    sc = AngularFourierSeries(bins = bins)

    # Featurize
    features = sc.featurize(structure, idx)
    feature_labels = sc.feature_labels()

    return (features, feature_labels)



## Functions to calculate electronic features
# These functions take in a vasprun object and calculate various electronic features


def dos(vasprun):
    from matminer.featurizers.dos import DOSFeaturizer

    complete_dos = vasprun.complete_dos

    # Initialize the featurizer
    sc = DOSFeaturizer()

    # Featurize the structure
    features = sc.featurize(complete_dos)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

#Does not work for our system since it is metallic
def band_featurizer(vasprun):
    from matminer.featurizers.bandstructure import BandFeaturizer

    try:
        band_structure = vasprun.get_band_structure()
        bf = BandFeaturizer()
        features = bf.featurize(band_structure)
        labels = bf.feature_labels()
        return (features, labels)

    except ValueError as e:
        if "metallic" in str(e).lower():
            print("BandFeaturizer skipped – metallic system (no band gap).")
        else:
            print(f"BandFeaturizer failed: {e}")
        return ([], [])

##functions to calculate packing and therodynamical properties

def packing(composition):
    from matminer.featurizers.composition.packing import AtomicPackingEfficiency

    sc=AtomicPackingEfficiency()
    

    features=sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def Thermo(composition):
    from matminer.featurizers.composition.thermo import CohesiveEnergy
    #needs api if your pymatgen version is 2023 or later you need to change get_data function in ->
    # cohesiveenergy feturize to summary_search(formula=comp.reduced_composition)
    sc=CohesiveEnergy(mapi_key='wQupTm0ToX93mMyQDCYWdChEqTbeROzU')
    

    try:
        features = sc.featurize(composition)
        feature_labels = sc.feature_labels()
        return (features, feature_labels)
    except ValueError as e:
        if "No structure found in MP" in str(e):
            #print(f"Warning: {e}")
            return ([], [])
        else:
            raise

def ThermoMP(composition):
    from matminer.featurizers.composition.thermo import CohesiveEnergyMP
    #needs api if your pymatgen version is 2023 or later you need to change get_data function in ->
    # cohesiveenergy feturize to summary_search(formula=comp.reduced_composition)
    sc=CohesiveEnergyMP(mapi_key='wQupTm0ToX93mMyQDCYWdChEqTbeROzU')

    
    try:
        features = sc.featurize(composition)
        feature_labels = sc.feature_labels()
        return (features, feature_labels)
    except ValueError as e:
        if "No structure found in MP" in str(e):
            #print(f"Warning: {e}")  # or use logging.warning(e)
            return ([], [])
        else:
            raise  # re-raise unexpected exceptions

##Sofie added composition features
##Composition features
def miedema(composition):
    from matminer.featurizers.composition import Miedema
    
    # Fills missing values with the mean of that feature across all elements
    sc = Miedema(impute_nan=True)

    #comp = structure.composition
    features = sc.featurize(composition)
    labels = sc.feature_labels()
    return features, labels

def yang(composition):
    from matminer.featurizers.composition import YangSolidSolution

    # Fills missing values with the mean of that feature across all elements
    sc = YangSolidSolution(impute_nan=True)

    #comp = structure.composition
    features = sc.featurize(composition)
    labels = sc.feature_labels()
    return features, labels

def wen(composition):
    from matminer.featurizers.composition import WenAlloys

    # Fills missing values with the mean of that feature across all elements
    sc = WenAlloys(impute_nan=True)

    #comp = structure.composition
    features = sc.featurize(composition)
    labels = sc.feature_labels()
    return features, labels

def element_property(composition):
    from matminer.featurizers.composition import ElementProperty

    # Generate composition-based features using the Magpie preset.
    sc = ElementProperty.from_preset("magpie")
    
    #comp = structure.composition
    features = sc.featurize(composition)
    labels = sc.feature_labels()
    return features, labels

def meredig(composition):
    from matminer.featurizers.composition import Meredig

    sc = Meredig()

    #comp = structure.composition
    features = sc.featurize(composition)
    labels = sc.feature_labels()
    return features, labels

#Composition features Rasmus 
def ElementFraction(composition):
    from matminer.featurizers.composition import ElementFraction
    
    # Initialize the featurizer
    sc = ElementFraction()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def TMetalFraction(composition):
    from matminer.featurizers.composition import TMetalFraction
    
    # Initialize the featurizer
    sc = TMetalFraction()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def Stoichiometry(composition):
    from matminer.featurizers.composition import Stoichiometry
    
    # Initialize the featurizer
    sc = Stoichiometry()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def BandCenter(composition):
    from matminer.featurizers.composition import BandCenter
    
    # Initialize the featurizer
    sc = BandCenter()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def OxidationStates(composition):
    from matminer.featurizers.composition import OxidationStates
    
    # Initialize the featurizer
    sc = OxidationStates()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def IonProperty(composition):
    from matminer.featurizers.composition import IonProperty
    
    # Initialize the featurizer
    sc = IonProperty()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def ElectronAffinity(composition):
    from matminer.featurizers.composition import ElectronAffinity
    
    # Initialize the featurizer
    sc = ElectronAffinity()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def ElectronegativityDiff(composition):
    from matminer.featurizers.composition import ElectronegativityDiff
    
    # Initialize the featurizer
    sc = ElectronegativityDiff()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def AtomicOrbitals(composition):
    from matminer.featurizers.composition import AtomicOrbitals
    
    # Initialize the featurizer
    sc = AtomicOrbitals()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def ValenceOrbital(composition):
    from matminer.featurizers.composition import ValenceOrbital
    
    # Initialize the featurizer
    sc = ValenceOrbital()

    # Featurize the structure
    features = sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)


############# functions to conversion



def ConversionFeaturizer(structure):
    from  matminer.featurizers.conversions import ConversionFeaturizer

    sc=ConversionFeaturizer(target_col_id='composition', overwrite_data=False)
    #print(structure)
    df = pd.DataFrame({'structure': structure})
    print(df.head())
    features=sc.featurize_dataframe(df,'structure',ignore_errors=True)
    print(features)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)


def StrToComposition(composition):
    from matminer.featurizers.conversions import StrToComposition

    sc=StrToComposition()
    

    features=sc.featurize(composition)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def StructureToComposition(structure):
    from matminer.featurizers.conversions import StructureToComposition


    sc=StructureToComposition()
    

    features=sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)


def StructureToIStructure(structure):
    from matminer.featurizers.conversions import StructureToIStructure

    sc=StructureToIStructure()
    

    features=sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def JsonToObject(structure):
    from matminer.featurizers.conversions import JsonToObject

    sc=JsonToObject()
    

    features=sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def DictToObject(structure):
    from matminer.featurizers.conversions import DictToObject
    #needs api
    sc=DictToObject()
    

    features=sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def StructureToOxidStructure(structure):
    from matminer.featurizers.conversions import StructureToOxidStructure

    sc=StructureToOxidStructure()
    

    features=sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def DensityFeatures(structure):
    from matminer.featurizers.structure.order import DensityFeatures
    
    # Initialize the featurizer
    sc = DensityFeatures()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)
def ChemicalOrdering(structure):
    from matminer.featurizers.structure.order import ChemicalOrdering
    from pymatgen.analysis.local_env import VoronoiNN
    #switch to Minimumdistance in localenviorment for 2d monolayer
    # Initialize the featurizer
    sc = ChemicalOrdering()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def MaximumPackingEfficiency(structure):
    from matminer.featurizers.structure.order import MaximumPackingEfficiency
    
    # Initialize the featurizer
    sc = MaximumPackingEfficiency()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def StructuralComplexity(structure):
    from matminer.featurizers.structure.order import StructuralComplexity
    
    # Initialize the featurizer
    sc = StructuralComplexity()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def RadialDistributionFunction(structure):
    from matminer.featurizers.structure.rdf import RadialDistributionFunction
    
    # Initialize the featurizer
    sc = RadialDistributionFunction()

    # Featurize the structure
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def PartialRadialDistributionFunction(structure):
    from matminer.featurizers.structure.rdf import PartialRadialDistributionFunction
    # Initialize the featurizer
    sc = PartialRadialDistributionFunction()

    # Featurize the structure
    sc.fit([structure])
    features = sc.featurize(structure)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def ElectronicRadialDistributionFunction(structure):
    from matminer.featurizers.structure.rdf import ElectronicRadialDistributionFunction

    #only works if the structure has oxidation_states
    # Initialize the featurizer
    sc = ElectronicRadialDistributionFunction()

    # Featurize the structure
    features = sc.featurize()
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

######################### Vilmer's functions
################ Functions to calculate conversion features
# different sorts of conversions

def CompositionToStructureFromMP(conversions):
    from matminer.featurizers.conversions import CompositionToStructureFromMP

    # Initialize the featurizer
    sc = CompositionToStructureFromMP()

    # Featurize the conversions
    features = sc.featurize(conversions)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def PymatgenFunctionApplicator(conversions):
    from matminer.featurizers.conversions import PymatgenFunctionApplicator
    
    # Initialize the featurizer
    sc = PymatgenFunctionApplicator(func=lambda x: x.volume)
    # lambda to return a value from a function that calls only functions
    #also works with x.density, x.composition.num_atoms and x.formula

    # Featurize the conversions
    features = sc.featurize(conversions)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def ASEAtomsToStructure(structure):
    from matminer.featurizers.conversions import ASEAtomstoStructure
    from pymatgen.io.ase import AseAtomsAdaptor
    from matminer.featurizers.structure import GlobalSymmetryFeatures

    # Convert Pymatgen Structure to ASE Atoms
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)

    # Use featurize to get the converted structure back from ASE atoms
    sc = ASEAtomstoStructure()
    converted_structure = sc.featurize(ase_atoms)[0]  # featurize returns a list

    # Now apply another featurizer (e.g., symmetry)
    symmetry_featurizer = GlobalSymmetryFeatures()
    features = symmetry_featurizer.featurize(converted_structure)
    feature_labels = symmetry_featurizer.feature_labels()

    return features, feature_labels


############### Functions to calculate DOS features
# gives information about density of states based on vasprun and, for SiteDOS, also an index for the sites

def SiteDOS(dos_vasp, idx=3):
    #idx is an index that can go between 0 and 3 in our relevant case
    from matminer.featurizers.dos import SiteDOS

    # Initialize the featurizer
    sc = SiteDOS()

    # Featurize the DOS
    features = sc.featurize(dos_vasp, idx)
    feature_labels = sc.feature_labels()
    
    return (features, feature_labels)

def DOSFeaturizer(dos_vasp):
    from matminer.featurizers.dos import DOSFeaturizer

    # Initialize the featurizer
    sc = DOSFeaturizer()

    # Featurize the DOS
    features = sc.featurize(dos_vasp)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def DopingFermi(dos_vasp):
    from matminer.featurizers.dos import DopingFermi

    # Initialize the featurizer
    sc = DopingFermi()

    # Featurize the DOS
    features = sc.featurize(dos_vasp)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def Hybridization(dos_vasp):
    from matminer.featurizers.dos import Hybridization
    
    # Initialize the featurizer
    sc = Hybridization()

    # Featurize the DOS
    features = sc.featurize(dos_vasp)
    feature_labels = sc.feature_labels()
    return (features, feature_labels)

def DosAsymmetry(dos_vasp):
    from matminer.featurizers.dos import DosAsymmetry

    sc = DosAsymmetry()
    feature = sc.featurize(dos_vasp)  # This is a float, big nono for our featurizer
    feature_label = ["DOS Asymmetry"]
    return ([feature], feature_label)  # Make it a list to stay consistent


############### Function to calculate function features
# Gets functions from features of other functions

def FunctionFeaturizer(function_values):
    import pandas as pd
    from matminer.featurizers.function import FunctionFeaturizer

    # Convert function_values into a DataFrame with column 'x'
    df = pd.DataFrame({"x": function_values})

    # Initialize and apply featurizer
    sc = FunctionFeaturizer()
    sc.fit(df)
    features_list = sc.transform(df)  # This returns a list of feature lists

    # Extract from the first row (only one row input)
    features = features_list[0]

    # Get the feature labels manually from the featurizer
    feature_labels = sc.feature_labels()

    return (features, feature_labels)

