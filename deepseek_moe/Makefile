hbm_pld:
	cd preload; python hbm_data_MoE.py

# hbm_pld_gemv:
# 	cd preload; python hbm_data_gemv.py
	
hbm_pld_decode:
	cd preload; python hbm_data_decode.py

hbm_pld_opt:
	cd preload; python hbm_data_opt.py

hw_512:
	cfg=$(abspath config/arch_NoC512.py) make -C ../ hw

hw_spatz:
	cfg=$(abspath config/arch_NoC512_spatz.py) make -C ../ hw

hw_16hbm:
	cfg=$(abspath config/arch_hbm16_NoC512.py) make -C ../ hw

# Bypass prelaod to save time
hs_moe:	
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs run

sw_moe:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ sw run

hs_moe_pfto: 
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs runv pfto

sw_moe_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ sw runv pfto

# Full run with preload
hs_moe_pld: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs run

sw_moe_pld: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ sw run

hs_moe_pld_pfto: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ hs runv pfto

sw_moe_pld_pfto: hbm_pld
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath applications) pld=$(abspath preload/hbm_data_MoE.elf) make -C ../ sw runv pfto

# Decode

hs_decode:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_decode.elf) make -C ../ hs run

hs_decode_opt:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_opt.elf) make -C ../ hs run

sw_decode:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_decode.elf) make -C ../ sw run

sw_decode_opt:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_opt.elf) make -C ../ sw run

hs_decode_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_opt.elf) make -C ../ hs runv pfto

sw_decode_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/hbm_data_opt.elf) make -C ../ sw runv pfto

hs_decode_clean:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) make -C ../ hs run

hs_decode_clean_spatz:
	cfg=$(abspath config/arch_NoC512_spatz.py) app=$(abspath moe_decode) make -C ../ hs run

hs_decode_clean_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) make -C ../ hs runv pfto

sw_decode_clean:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) make -C ../ sw run

sw_decode_clean_spatz:
	cfg=$(abspath config/arch_NoC512_spatz.py) app=$(abspath moe_decode) make -C ../ sw run

sw_decode_clean_spatz_pfto:
	cfg=$(abspath config/arch_NoC512_spatz.py) app=$(abspath moe_decode) make -C ../ sw runv pfto

sw_decode_clean_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) make -C ../ sw runv pfto

# Centralized decode
sw_decode_centralized:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/preload.elf) make -C ../ sw run

sw_decode_centralized_pfto:
	cfg=$(abspath config/arch_NoC512.py) app=$(abspath moe_decode) pld=$(abspath preload/preload.elf) make -C ../ sw runv pfto
	

# GEMV
# hbm_gemv:
# 	cd preload; python hbm_data_gemv.py

# hs_gemv_pfto:
# 	cfg=$(abspath config/arch_NoC512.py) app=$(abspath gemv) pld=$(abspath preload/hbm_data_gemv.elf) make -C ../ hs runv pfto

# hs_gemv_pld_pfto: hbm_gemv
# 	cfg=$(abspath config/arch_NoC512.py) app=$(abspath gemv) pld=$(abspath preload/hbm_data_gemv.elf) make -C ../ hs runv pfto

# sw_gemv_pfto:
# 	cfg=$(abspath config/arch_NoC512.py) app=$(abspath gemv) pld=$(abspath preload/hbm_data_gemv.elf) make -C ../ sw runv pfto