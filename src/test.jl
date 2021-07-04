function tester(b = true)
  if b
    x = false
  else
    x = nothing
  end

  for _ = 1:3
    x = tester(x)
  end
  return x
end
tester(::Nothing) = nothing
