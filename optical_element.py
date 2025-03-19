def phase2height(phase_u: torch.Tensor, wvl: float, RI: float, minh: float = 0) -> torch.Tensor:
    """Convert phase change to material height.

    Note that phase to height mapping is not one-to-one.
    There exists an integer phase wrapping factor:
        height = wvl/(RI-1) * (phase_u + i*2π), where i is integer
    This function uses minimum height minh to constrain the conversion.
    Minimal height is chosen such that height is always >= minh.

    Args:
        phase_u (torch.Tensor): Phase change of light
        wvl (float): Wavelength of light in meters
        RI (float): Refractive index of material at given wavelength
        minh (float): Minimum height constraint in meters

    Returns:
        torch.Tensor: Material height that induces the phase change

    Examples:
        >>> phase = torch.ones((1,1,1024,1024)) * np.pi
        >>> height = phase2height(phase, 633e-9, 1.5, minh=100e-9)  # 100nm min height
    """ 