#!/bin/bash
{   
    export CUDA_VISIBLE_DEVICES=0,1
    METHOD=PN
    TASK=5shot
    TASKID=PN_codebase
    WANDB_PROJECT=FSLdoing
    CONFIG_FILE=config/config.yaml
    CONFIG_PY=config/set_config_PN_train.py

    git_status="$(git status 2> /dev/null)"
    branch_pattern="^(# )?On branch ([^${IFS}]*)"
    if [[ ${git_status} =~ ${branch_pattern} ]]; then
        BRANCH=${BASH_REMATCH[2]}
    fi

    ckpt_dir=results/${BRANCH}/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}

    mkdir -p ${ckpt_dir}

    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT}
    cp ${CONFIG_FILE} ${ckpt_dir}
    cp train.sh ${ckpt_dir}
    python run.py --config ${CONFIG_FILE}
    exit
}