def NMC811_KaranthWeijers24(y):
    """Fitted from the results of McClelland 2023"""
    # Implemented by Ombrini for Karanth and Weijers 2024
    # Diffusion coefficient in cm^2/s
    # Ranges from 1e-11 to 2e-10
    Diff = 0.5*10**(-66631.56720002113*y**9 + 317224.13759806077*y**8 
                    + -647127.9100798424*y**7 + 740625.6188941287*y**6 
                    + -522889.4946864639*y**5 + 235652.79793610598*y**4 
                    + -67638.17186534218*y**3 + 11887.013361406942*y**2 
                    + -1155.8947947859672*y + 37.60120496926822)
    return Diff/1e4
