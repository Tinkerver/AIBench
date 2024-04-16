export INSTALL_DIR=~/Ascend/ascend-toolkit/latest
export DDK_PATH=${INSTALL_DIR}
export NPU_HOST_LIB=${INSTALL_DIR}/x86_64-linux/devlib
export LD_LIBRARY_PATH=${INSTALL_DIR}/x86_64-linux/devlib/:$LD_LIBRARY_PATH

for ((i=1; i<=100; i++))
do
	echo $i
	# rm /home/tinker-910/MindstudioProjects/untitled1/tbe/impl/tg.py
	python3 /home/tinker-910/test/vector_workload_generate.py $i /home/tinker-910/MindstudioProjects/untitled1/tbe/impl/tg.py

	cd /home/tinker-910/MindstudioProjects/untitled1/
	/home/tinker-910/MindstudioProjects/untitled1/build.sh clean
	/home/tinker-910/MindstudioProjects/untitled1/build.sh
	/home/tinker-910/MindstudioProjects/untitled1/build_out/custom_opp_ubuntu_x86_64.run

	~/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopst run \
	-i /home/tinker-910/MindstudioProjects/untitled1/testcases/st/tg/ascend310/Tg_case_20240108222840.json \
	-soc Ascend310 \
	-out /home/tinker-910/out/test_$i \
	-conf ~/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopst.ini
done

# sshpass -p 123456 scp -r -P 3022 /home/tinker-910/out/test_* root@127.0.0.1:/root/out_main/test
