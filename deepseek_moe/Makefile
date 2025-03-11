hbm_pld:
	cd preload; python hbm_data_MoE.py

# Bypass prelaod to save time
hs_moe:	
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs run

sw_moe:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ sw run

hs_moe_pfto: 
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs runv pfto

# Full run with preload
hs_pld: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs run

hs_pld_pfto: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs runv pfto