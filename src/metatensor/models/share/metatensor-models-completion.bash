_metatensor-models()
{
  local cur_word="${COMP_WORDS[$COMP_CWORD]}"
  local prev_word="${COMP_WORDS[$COMP_CWORD-1]}"
  local module="${COMP_WORDS[1]}"

  # Define supported file endings.
  local yaml='!*@(.yml|.yaml)'
  local ckpt='!*.ckpt'
  local pt='!*.pt'

  # Complete the arguments to the module commands.
  case "$module" in
    train)
      case "${prev_word}" in
        -h|--help)
          COMPREPLY=( )
          return 0
          ;;
        -o|--output)
          COMPREPLY=( )
          return 0
          ;;
        -c|--continue)
          COMPREPLY=( $( compgen -f -X "$ckpt" -- "${cur_word}") )
          return 0
          ;;
        -r|--override)
          COMPREPLY=( )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -f -X "$yaml" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output -c --continue -r --override"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
    export)
      case "${prev_word}" in
        -o|--output)
          COMPREPLY=( )
          return 0
          ;;
        -h|--help)
          COMPREPLY=( )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -f -X "$ckpt" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
    eval)
      case "${prev_word}" in
        -o|--output)
          COMPREPLY=( )
          return 0
          ;;
        -h|--help)
          COMPREPLY=( )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -f -X "$pt" -- "${cur_word}") )
            return 0
          elif [[ $COMP_CWORD -eq 3 ]]; then
            COMPREPLY=( $(compgen -f -X "$yaml" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
  esac

  # Complete the basic metatensor-models commands.
  local opts="eval export train -h --help --debug --version"
  COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
  return 0
}

if test -n "$ZSH_VERSION"; then
  autoload -U +X compinit && compinit
  autoload -U +X bashcompinit && bashcompinit
fi

complete -o bashdefault -F _metatensor-models metatensor-models
